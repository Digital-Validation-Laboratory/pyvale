from pathlib import Path
import mooseherder as mh

def main() -> None:
    gmsh_path = Path().home() / 'moose-workdir/gmsh/bin/gmsh'
    gmsh_runner = mh.GmshRunner(gmsh_path)


    base_dir = Path('dev/tsr-dev/gmsh')
    gmsh_in = base_dir / 'claude.geo'
    #gmsh_in = base_dir / 'gmsh_t2.geo'
    gmsh_runner.set_input_file(gmsh_in)

    gmsh_runner.run()

if __name__ == '__main__':
    main()