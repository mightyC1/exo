import faulthandler
import os
import resource
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Self, cast

import loguru

from exo.shared.types.events import Event
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.base import Builder

logger: "loguru.Logger" = loguru.logger


@dataclass(frozen=True)
class RunnerTerminationError:
    exception_type: str
    exception_message: str
    exception_repr: str
    traceback: str

    @classmethod
    def from_exception(cls, e: Exception) -> Self:
        return cls(
            exception_type=type(e).__qualname__,
            exception_message=str(e),
            exception_repr=repr(e),
            traceback="".join(
                traceback.TracebackException.from_exception(e).format(chain=True)
            ),
        )

    def __str__(self) -> str:
        return f"{self.exception_type}: {self.exception_message}\n{self.traceback}"


_CRASH_LOG = None  # держим хэндл живым до конца процесса


def _enable_crash_capture(runner_id: object) -> None:
    """faulthandler в файл: при SIGSEGV/SIGABRT/SIGBUS (Metal/jaccl C++)
    дампит python-стеки ВСЕХ тредов туда, где их можно найти после смерти —
    в отличие от tmux-скроллбека. ml-explore/mlx#3207-класс отладки."""
    global _CRASH_LOG
    try:
        crash_dir = Path.home() / ".exo" / "crash"
        crash_dir.mkdir(parents=True, exist_ok=True)
        path = crash_dir / f"runner-{runner_id}-{os.getpid()}-{int(time.time())}.log"
        _CRASH_LOG = open(path, "w", buffering=1)
        _CRASH_LOG.write(f"runner={runner_id} pid={os.getpid()} started\n")
        faulthandler.enable(file=_CRASH_LOG, all_threads=True)
        logger.info(f"crash capture -> {path}")
    except Exception as e:  # диагностика не должна ронять раннер
        logger.warning(f"crash capture disabled: {e}")


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event | RunnerTerminationError],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    _enable_crash_capture(getattr(bound_instance, 'bound_runner_id', 'unknown'))
    global logger
    logger = _logger

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override == "false":
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        event_sender_downcast: MpSender[Event] = cast(MpSender[Event], event_sender)

        from exo.worker.runner.runner import Runner

        builder: Builder
        if bound_instance.is_image_model:
            from exo.worker.engines.image.builder import MfluxBuilder

            builder = MfluxBuilder(
                event_sender_downcast, cancel_receiver, bound_instance.bound_shard
            )
        else:
            from exo.worker.engines.mlx.patches import apply_mlx_patches

            apply_mlx_patches()

            from exo.worker.engines.mlx.builder import MlxBuilder

            # evil sharing of the event sender
            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender_downcast,
                cancel_receiver=cancel_receiver,
            )

        runner = Runner(bound_instance, builder, event_sender_downcast, task_receiver)
        from exo.worker.engines.mlx.patches.prefix_flush import install as _pf_install
        _pf_install()
        runner.main()
    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(RunnerTerminationError.from_exception(e))
        # Crash-path teardown: the exception traceback pins the whole frame
        # stack (locals in mlx/generator frames keep the JACCL group alive),
        # so refcounts never fall before the supervisor kills the process and
        # the kernel is left with an un-destroyed RDMA context (node wedge).
        # Best effort: close the generator, drop the pins, collect, and only
        # then exit. If the Group destructor hangs on in-flight collectives,
        # the supervisor kill remains the backstop (no worse than today).
        try:
            _r = None
            try:
                _r = runner  # may be unbound if crash predates Runner()
            except NameError:
                pass
            if _r is not None and getattr(_r, "generator", None) is not None:
                try:
                    _r.generator.close()
                except Exception:
                    logger.warning("crash-path generator.close() failed")
                _r.generator = None  # type: ignore[assignment]
            try:
                from exo.worker.engines.mlx.auto_parallel import (
                    clear_prefill_sends,
                )

                clear_prefill_sends()
            except Exception:
                pass
            e.__traceback__ = None  # release frame pins
            import gc

            gc.collect()
        except Exception:
            logger.warning("crash-path teardown failed")
        raise SystemExit(1) from e
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
