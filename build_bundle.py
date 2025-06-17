from pathlib import Path

from chimerax.core.commands.devel import devel_clean, devel_build
from chimerax.core.commands.toolshed import toolshed_install

bundle_name = "volume_distance_cmd"
bundle_path = (Path(__file__).parent / bundle_name).resolve()

if "dist" in bundle_path.glob("*"):
    devel_clean(session, str(bundle_path))
devel_build(session, str(bundle_path))

wheel_path = list(map(str, list((bundle_path / "dist").glob("*.whl"))))
toolshed_install(session, wheel_path, reinstall=False)  # type: ignore
