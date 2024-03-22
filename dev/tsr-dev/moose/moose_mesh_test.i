# Moose Mesh Test Script

[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    type = FileMesh
    file = 'mesh_3d_monoblock.msh'
[]

[Modules/TensorMechanics/Master]
    [all]
        add_variables = true
    []
[]

[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = 1e9
        poissons_ratio = 0.3
    []
    [stress]
        type = ComputeLinearElasticStress
    []
[]

[Executioner]
    type = Transient
    end_time = 2
    dt = 1
[]

[Outputs]
    exodus = true
[]
