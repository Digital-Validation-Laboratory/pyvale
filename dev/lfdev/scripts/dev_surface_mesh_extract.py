
from pathlib import Path
import mooseherder as mh
import pyvale

def main() -> None:
    data_path = Path("src/pyvale/simcases/case21_out.e")
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()

    (pv_grid,pv_grid_vis) = pyvale.conv_simdata_to_pyvista(sim_data,
                                                           None,
                                                           spat_dim=3)

    pv_surf = pv_grid.extract_surface()

    save_path = Path().cwd() / "test_output"
    if not save_path.is_dir():
        save_path.mkdir()
    save_file = save_path / "test_mesh.stl"

    pv_surf.save(save_file,binary=False)


if __name__ == "__main__":
    main()