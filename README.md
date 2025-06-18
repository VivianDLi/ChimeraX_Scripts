# Usage
## Installing files
On the GitHub page, copy the link found in the green **Code** button under HTTPS (or SSH if you have that set-up).
Then, in terminal, go to the folder where you want to have the script files.

Run `git clone <copied_url>`.

## Installing bundle
After opening ChimeraX, run in the command-line `open <path_to_script_files>/build_bundle.py`.
This should build the bundle (a `.whl` file in the `/dist` directory), and install it into ChimeraX.

If any of the source code is changed, then change `reinstall=False` to `reinstall=True` in `build_bundle.py`, and running the `open` command above should work to update the bundle after restarting ChimeraX.
Note: running `toolshed uninstall VolumeDistanceCommand` and re-running the `open` command might also work, but I've found restarting is more consistent.

## Running commands
The bundle loads three commands:

`volume distance single`
This command is for calculating the distances between two individual models (either a Surface or a Volume) or between a model and itself.
  For comparing two models, use the `to` argument to specify the second.

`volume distance multi`
This command is for calculating the distances between several individual models. This may be useful for studying a subset of models.
  For comparing two separate sets of models, use the `to` argument to specify the second.

`volume distance group`
This command is for calculating the distances between models, where the models are grouped under a parent model in ChimeraX.

All three commands have the same optional keyword arguments:
  For general use, these arguments are the only ones that should be touched:
    `cluster_angle_threshold` - the distance calculations uses a **region-growing** algorithm to separate out individual structures from a model, and these are shown as a separate model with separate colors per structure. Adjust this parameter (default=0.785 or 45 degrees) to change how this works, where decreasing it will produce more structures, and increasing it will produce less.
      This is essentially the threshold for the surface normal angle diffference between two points for a point to be added to the same structure as the other point.
    `bond_radius` - how thick distances (bonds) appear in ChimeraX.
    `use_surface` - if the model already has a surface calculated for it (in ChimeraX), enabling this (enabled by default) speeds up the calculations. Otherwise, the code will calculate the surface points from the raw volume data.
    `use_internal` - if the distance calculations should use an internal ChimeraX function. This is disabled by default.
    `use_mean` - if the distance calculations should use the mean of source surface points to calculate distance with. Enabling this speeds up the calculation. For small objects (e.g., nuclear pores), this doesn't affect the result much, but for larger objects (e.g., mitochondria), I would recommend setting this to false (`use_mean False`). This is enabled by default.

  These arguments are for changing the behaviour of internal algorithms and shouldn't be touched unless you have knowledge of the algorithms used:
    `surface_radius` - this defines the radius to search for neighbours when estimating normals (using eigenvalue decomposition). this is used for calculating surface points from volume data.
    `surface_point_tol` - this defines the threshold number of points found in either normal direction for a point to be considered a surface point. this is used for calculating surface points from volume data.
    `cluster_normal_k` - this defines the number of nearest neighbours to find when estimating normals (using eigenvalue decomposition). this is used for finding distinct structures with region-growing.
    `cluster_region_k` - this defines the number of nearest neighbours to find around each seed in the region-growing algorithm.
    `cluster_curv_threshold` - this is the curvature threshold for additionally filtering candidate structure points in the region-growing algorithm. this is default set to 1.0 (no filter).

## Examples
Mitochondria dataset:
`volume distance group #1 use_mean False`

Cancer dataset:
`volume distance single #6 to #14`
