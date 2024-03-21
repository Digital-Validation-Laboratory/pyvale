from pathlib import Path
import time
import mooseherder as mh

def main() -> None:
    gmsh_path = Path().home() / 'moose-workdir/gmsh/bin/gmsh'
    gmsh_runner = mh.GmshRunner(gmsh_path)


    base_dir = Path('dev/tsr-dev/gmsh')
    #gmsh_in = base_dir / 'gmsh_3d_plate.geo'
    gmsh_in = base_dir / 'gmsh_3d_monoblock.geo'
    #gmsh_in = base_dir / 'gmsh_t3.geo'
    gmsh_runner.set_input_file(gmsh_in)

    start_time = time.perf_counter()
    gmsh_runner.run()
    run_time = time.perf_counter() - start_time

    print()
    print("="*80)
    print(f'Gmsh run time = {run_time:.3f} seconds')
    print("="*80)
    print()


if __name__ == '__main__':
    main()