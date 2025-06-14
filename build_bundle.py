from pathlib import Path

from chimerax.core.commands.devel import devel_clean, devel_build
from chimerax.core.commands.toolshed import toolshed_install

bundle_name = "volume_distance_cmd"
bundle_path = Path(f"./{bundle_name}").resolve()

if "dist" in bundle_path.glob("*"):
    devel_clean(str(bundle_path))
devel_build(str(bundle_path))

wheel_path = list((bundle_path / "dist").glob("*.whl"))[0]
toolshed_install(session, wheel_path, reinstall=False)  # type: ignore
