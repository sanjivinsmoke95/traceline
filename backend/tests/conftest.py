"""Test fixtures. The suite runs against TEST_DATABASE_URL only, never the development database."""
import argparse
import subprocess
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.api.deps import get_factory
from app.config import get_settings
from app.db import get_db
from app.main import app

BACKEND = Path(__file__).resolve().parents[1]


def _alembic_cfg(url: str) -> Config:
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "alembic"))
    cfg.cmd_opts = argparse.Namespace(x=[f"url={url}"])
    return cfg


def _truncate_all(engine) -> None:
    """TRUNCATE bypasses the row-level append-only guards, which is what a test reset needs."""
    with engine.begin() as conn:
        tables = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename <> 'alembic_version'")).scalars().all()
        conn.execute(text("TRUNCATE " + ", ".join(tables) + " RESTART IDENTITY CASCADE"))


@pytest.fixture(scope="session")
def test_url() -> str:
    url = get_settings().test_database_url
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    assert url != get_settings().database_url, "refusing to run tests against the development database"
    return url


@pytest.fixture(scope="session")
def engine(test_url):
    cfg = _alembic_cfg(test_url)
    eng = create_engine(test_url, future=True)
    with eng.begin() as conn:
        tables = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename <> 'alembic_version'")).scalars().all()
        if tables:
            conn.execute(text("TRUNCATE " + ", ".join(tables) + " RESTART IDENTITY CASCADE"))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    _truncate_all(engine)
    with Session(engine, expire_on_commit=False) as s:
        yield s
        s.rollback()


@pytest.fixture()
def client(engine, tmp_path):
    """API client wired to the test database and a throwaway evidence store, with small upload limits."""
    from app.config import get_settings as settings_dep

    _truncate_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    real = settings_dep()
    settings = real.model_copy(update={"storage_root": tmp_path / "store", "max_video_mb": 5, "max_other_mb": 1,
                                       "models_dir": real.models_path,
                                       "accident_backend": "yolo"})  # pinned weights are shared, read-only; the DETR cue (a ~160 MB download) is tested separately with a stub

    def db_dep():
        with factory() as s:
            yield s

    app.dependency_overrides[get_db] = db_dep
    app.dependency_overrides[settings_dep] = lambda: settings
    app.dependency_overrides[get_factory] = lambda: factory
    c = TestClient(app)
    c.settings, c.factory = settings, factory
    yield c
    app.dependency_overrides.clear()


def _ffmpeg(args: list[str], out: Path) -> Path:
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", *args, str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """2 s, 320x240, 15 fps H.264 test pattern."""
    d = tmp_path_factory.mktemp("video")
    return _ffmpeg(["-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15", "-pix_fmt", "yuv420p", "-c:v", "libx264"], d / "ok.mp4")


@pytest.fixture(scope="session")
def gappy_video(tmp_path_factory) -> Path:
    """4 s at 15 fps with frames 20-35 removed while keeping the original timestamps (about a 1 s gap)."""
    d = tmp_path_factory.mktemp("gappy")
    return _ffmpeg(["-f", "lavfi", "-i", "testsrc=duration=4:size=320x240:rate=15", "-vf", "select='not(between(n,20,35))'",
                    "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-c:v", "libx264"], d / "gappy.mp4")


@pytest.fixture(scope="session")
def offset_video(tmp_path_factory) -> Path:
    """2 s at 15 fps whose first frame is stamped ~2.5 s into the container timeline."""
    d = tmp_path_factory.mktemp("offset")
    return _ffmpeg(["-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15", "-vf", "setpts=PTS+2.5/TB",
                    "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", "-c:v", "libx264"], d / "offset.mp4")


@pytest.fixture(scope="session")
def mkv_video(tmp_path_factory) -> Path:
    """MPEG-4 Part 2 in Matroska: not playable in browsers, so it must be transcoded."""
    d = tmp_path_factory.mktemp("mkv")
    return _ffmpeg(["-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15", "-c:v", "mpeg4"], d / "cam.mkv")


MODELS = get_settings().models_path
DATASET = get_settings().storage_path / "datasets/accident_detection/valid"
needs_weights = pytest.mark.skipif(not (MODELS / "yolo26s.pt").is_file(), reason="detector weights not installed (scripts/fetch_models.py)")
needs_accident = pytest.mark.skipif(not (MODELS / "accident_yolov8s_best.pt").is_file() or not DATASET.is_dir(),
                                    reason="accident model or its validation split not installed")


@pytest.fixture(scope="session")
def bus_video(tmp_path_factory) -> Path:
    """3 s pan over Ultralytics' bundled street photo (a bus and people), 10 fps."""
    import ultralytics

    src = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    d = tmp_path_factory.mktemp("bus")
    return _ffmpeg(["-loop", "1", "-i", str(src), "-vf", "crop=640:480:'t*40':300", "-t", "3", "-r", "10",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264"], d / "bus.mp4")


@pytest.fixture(scope="session")
def accident_video(tmp_path_factory) -> Path:
    """2 s still clip of a validation image on which the accident model fires confidently on a labelled box."""
    from app.services.weights import load_yolo

    model, _ = load_yolo("accident_yolov8s", str(MODELS))
    chosen = None
    for img in sorted((DATASET / "images").iterdir()):
        if (DATASET / "labels" / (img.stem + ".txt")).read_text().strip():
            r = model.predict(str(img), conf=0.6, verbose=False, device="cpu")[0]
            if len(r.boxes):
                chosen = img
                break
    assert chosen is not None, "no confidently detected validation image found"
    d = tmp_path_factory.mktemp("acc")
    return _ffmpeg(["-loop", "1", "-i", str(chosen), "-vf", "scale=640:-2", "-t", "2", "-r", "5", "-pix_fmt", "yuv420p", "-c:v", "libx264"], d / "acc.mp4")
