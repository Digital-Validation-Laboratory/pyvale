"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
from pathlib import Path
from importlib.resources import files


SIM_CASE_COUNT = 24


class DataSetError(Exception):
    """Custom error class for file io errors associated with retrieving datasets
    and files packaged with pyvale.
    """
    pass


class DataSet:
    """A static namespace class for handling datasets packaged with pyvale.
    Contains a series of static methods returning a Path object to each data
    file that is packaged with pyvale.
    """

    @staticmethod
    def sim_case_input_file_path(case_num: int) -> Path:
        """Gets the path to MOOSE input file (*.i) for a particular simulation
        case.

        Parameters
        ----------
        case_num : int
            Integer defining the case number to be retrieved. Must be greater
            than 0 and less than the number of simulation cases.

        Returns
        -------
        Path
            Path object to the MOOSE *.i file for the selected simulation case.

        Raises
        ------
        DataSetError
            Raised if an invalid simulation case number is specified.
        """
        if case_num <= 0:
            raise DataSetError("Simulation case number must be greater than 0")
        elif case_num > SIM_CASE_COUNT:
            raise DataSetError("Simulation case number must be less than " \
                               + f"{SIM_CASE_COUNT}")

        case_num_str = str(case_num).zfill(2)
        case_file = f"case{case_num_str}.i"
        return Path(files("pyvale.simcases").joinpath(case_file))


    @staticmethod
    def sim_case_gmsh_file_path(case_num: int) -> Path | None:
        """Gets the path to Gmsh input file (*.geo) for a particular simulation
        case. Note that not all simulation cases use Gmsh for geometry and mesh
        generation. If the specified simulation case does not have an associated
        Gmsh *.geo file. In this case 'None' is returned

        Parameters
        ----------
        case_num : int
            Integer defining the case number to be retrieved. Must be greater
            than 0 and less than the number of simulation cases.

        Returns
        -------
        Path | None
            Path object to the Gmsh *.geo file for the selected simulation case.
            Returns None if there is no *.geo for this simulation case.

        Raises
        ------
        DataSetError
            Raised if an invalid simulation case number is specified.
        """
        if case_num <= 0:
            raise DataSetError("Simulation case number must be greater than 0")
        elif case_num > SIM_CASE_COUNT:
            raise DataSetError("Simulation case number must be less than " \
                               + f"{SIM_CASE_COUNT}")

        case_num_str = str(case_num).zfill(2)
        case_file = f"case{case_num_str}.geo"
        case_path = Path(files("pyvale.simcases").joinpath(case_file))

        if case_path.is_file():
            return case_path

        return None


    @staticmethod
    def dic_pattern_5mpx_path() -> Path:
        """Path to a 5 mega-pixel speckle pattern image (2464 x 2056 pixels)
        with 8 bit resolution stored as a *.tiff. Speckles are sampled by
        5 pixels. A gaussian blur has been applied to the image to remove sharp
        transitions from black to white.

        Path
            Path to the *.tiff file containing the speckle pattern.
        """
        return Path(files("pyvale.data")
                    .joinpath("optspeckle_2464x2056px_spec5px_8bit_gblur1px.tiff"))

    @staticmethod
    def thermal_2d_output_path() -> Path:
        """Path to a MOOSE simulation output in exodus format. This case is a
        thermal problem solving for a scalar temperature field. The geometry is
        a 2D plate (in x,y) with a heat flux applied on one edge and a heat
        transfer coefficient applied on the opposite edge inducing a temperature
        gradient along the x axis of the plate.

        The simulation parameters can be found in the corresponding MOOSE input
        file: case13.i which can be retrieved using `sim_ca_summary_

    Parameters
    ----------
    Exception : _type_
        _description_se_input_file_path`
        in this class.

        Returns
        -------
        Path
            Path to the exodus (*.e) output file for this simulation case.
        """
        return Path(files("pyvale.data").joinpath("case13_out.e"))

    @staticmethod
    def thermal_3d_output_path() -> Path:
        """Path to a MOOSE simulation output in exodus format. This case is a 3D
        thermal problem solving for a scalar temperature field. The model is a
        divertor armour monoblock composed of a tungsten block bonded to a
        copper-chromium-zirconium pipe with a pure copper interlayer. A heat
        flux is applied to the top surface of the block and a heat transfer
        coefficient for cooling water is applied to the inner surface of the
        pipe inducing a temperature gradient from the top of the block to the
        pipe.

        The simulation parameters can be found in the corresponding MOOSE input
        file: case16.i which can be retrieved using `sim_case_input_file_path`
        in this class. Note that this case uses a Gmsh *.geo file for geometry
        and mesh creation.

        Returns
        -------
        Path
            Path to the exodus (*.e) output file for this simulation case.
        """
        return Path(files("pyvale.data").joinpath("case16_out.e"))

    @staticmethod
    def mechanical_2d_output_path() -> Path:
        """Path to a MOOSE simulation output in exodus format. This case is a 2D
        plate with a hole in the center with the bottom edge fixed and a
        displacement applied to the top edge. This is a mechanical problem and
        solves for the displacement vector field and the tensorial strain field.

        The simulation parameters can be found in the corresponding MOOSE input
        file: case17.i which can be retrieved using `sim_case_input_file_path`
        in this class. Note that this case uses a Gmsh *.geo file for geometry
        and mesh creation.

        Returns
        -------
        Path
            Path to the exodus (*.e) output file for this simulation case.
        """
        return Path(files("pyvale.data").joinpath("case17_out.e"))

    @staticmethod
    def thermomechanical_2d_output_path() -> Path:
        """Path to a MOOSE simulation output in exodus format. This case is a
        thermo-mechanical analysis of a 2D plate with a heat flux applied on one
        edge and a heat transfer coefficient applied on the opposing edge. The
        mechanical deformation results from thermal expansion due to the imposed
        temperature gradient. This model is solved for the scalar temperature
        field, vector temperature and tensor strain field._summary_

    Parameters
    ----------
    Exception : _type_
        _description_responding MOOSE input
        file: case18.i which can be retrieved using `sim_case_input_file_path`
        in this class.

        Returns
        -------
        Path
            Path to the exodus (*.e) output file for this simulation case.
        """
        return Path(files("pyvale.data").joinpath("case18_1_out.e"))

    @staticmethod
    def thermomechanical_2d_experiment_output_paths() -> list[Path]:
        """Path to a MOOSE simulation output in exodus format. This case is a
        thermo-mechanical analysis of a 2D plate with a heat flux applied on one
        edge and a heat transfer coefficient applied on the opposing edge. The
        mechanical deformation results from thermal expansion due to the imposed
        temperature gradient. This model is solved for the scalar temperature
        field, vector temperature and tensor strain field.

        Here we analyse 3 separate experiments where the thermal conductivity of
        the material is perturbed from the nominal case by +/-10%.

        The simulation parameters can be found in the corresponding MOOSE input
        file: case18.i which can be retrieved using `sim_case_input_file_path`
        in this class.

        Returns
        -------
        Path
            Path to the exodus (*.e) output file for this simulation case.
        """
        return [Path(files("pyvale.data").joinpath("case18_1_out.e")),
                Path(files("pyvale.data").joinpath("case18_2_out.e")),
                Path(files("pyvale.data").joinpath("case18_3_out.e"))]

