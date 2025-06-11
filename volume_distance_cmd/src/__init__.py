from chimerax.core.toolshed import BundleAPI


# Subclass from chimerax.core.toolshed.BundleAPI and
# override the method for registering commands,
# inheriting all other methods from the base class.
class _MyAPI(BundleAPI):
    api_version = (
        1  # register_command uses BundleInfo and CommandInfo instead of command name
    )

    @staticmethod
    def register_command(bi, ci, logger):
        # Check the name of the command
        # and import function to call and argument description from cmd.py
        # If the description does not contain a 'synopsis', we use the one from 'ci'
        # taken from bundle_info.xml
        from .cmd import calculate_volume_distance, volume_distance_desc

        if ci.name == "volume distance":
            func = calculate_volume_distance
            desc = volume_distance_desc
        else:
            raise ValueError("Trying to register an unknown command: %s" % ci.name)
        if desc.synopsis is None:
            desc.synopsis = ci.synopsis

        # Register command
        from chimerax.core.commands import register

        register(ci.name, desc, func, logger=logger)


bundle_api = _MyAPI()
