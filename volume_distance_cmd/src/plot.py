"""Plot distances on a histogram (based on crosslinks.LengthsPlot)."""

from chimerax.interfaces.graph import Plot


class VolumeDistancePlot(Plot):
    """A plot for displaying distances between volumes."""

    def __init__(
        self, session, bonds, bins=50, max_length=None, min_length=None, height=None
    ):
        title = f"{len(bonds)} Distances"
        super().__init__(session, tool_name="Volume Distance", title=title)
        self.tool_window.fill_context_menu = self._fill_context_menu

        self.bonds = bonds
        self._bins = bins
        self._max_length = max_length
        self._min_length = min_length
        self._height = height

        self._bin_edges = None
        self._patches = []
        self._last_picked_bin = None
        self._fattened_bonds = None
        self._unfattened_radii = None

        self._make_histogram()
        self.figure.set_tight_layout(True)

        self.canvas.mouseMoveEvent = self._on_mouse_move

        self.show()

    def _make_histogram(self):
        a = self.axes
        a.set_title("Distances between volumes")
        a.set_xlabel(r"Distance ($\AA$)")
        a.set_ylabel("Counts")
        if self._height is not None:
            a.set_ylim([0, self._height])

        distances = self.bonds.lengths
        bins = self._bins

        value_range = [
            self._min_length or distances.min(),
            self._max_length or distances.max(),
        ]

        n, be, self._patches = a.hist(distances, bins=bins, range=value_range)
        self._bin_edges = be

        if (
            bins > 0
        ):  # Handle the edge == max value case making np.digitize ignore the max value
            be[bins] += 0.01 * (be[bins] - be[bins - 1])

        # Map bins to the bonds that fall into them
        from numpy import digitize

        bin_i = digitize(distances, self._bin_edges) - 1  # 0-indexed
        bin_mappings = {}
        for i, b in zip(bin_i, self.bonds):
            bin_mappings.setdefault(i, []).append(b)
        from chimerax.atomic import Bonds

        self._bin_bonds = {i: Bonds(bs) for i, bs in bin_mappings.items()}

    def _on_mouse_move(self, event):
        pos = event.pos()
        e = self.matplotlib_mouse_event(pos.x(), pos.y())
        # Enable mouse hover to highlight bonds in a bin
        for i, p in enumerate(self._patches):
            c, _ = p.contains(e)
            if c:
                self._pick_bin(i)
                return
        self._pick_bin(None)

    def _pick_bin(self, b):
        if (
            b == self._last_picked_bin
        ):  # Save processing time by not re-picking the same bin
            return

        self._last_picked_bin = b
        fattened_bonds = self._fattened_bonds
        if fattened_bonds is not None:
            # Reset highlighted bonds
            fattened_bonds.radii = self._unfattened_radii

        if b is None:
            self._fattened_bonds = None
            self._unfattened_radii = None
            self.bonds.displays = True
        else:
            bonds = self.bonds
            picked_bonds = self._bin_bonds.get(b, None)
            if picked_bonds is None or len(picked_bonds) == 0:
                return
            self._fattened_bonds = picked_bonds
            self._unfattened_radii = picked_bonds.radii
            picked_bonds.radii *= 3.0
            bonds.displays = False
            picked_bonds.displays = True

    def _fill_context_menu(self, menu):
        self.add_menu_entry(menu, "Save Plot As...", self.save_plot_as)
