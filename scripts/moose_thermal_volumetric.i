#
# Single block thermal input with time derivative and volumetric heat source terms
# https://mooseframework.inl.gov/modules/heat_transfer/tutorials/introduction/therm_step03.html
#

[Mesh]
    [generated]
        type = GeneratedMeshGenerator
        dim = 2
        nx = 10
        ny = 10
        xmax = 2
        ymax = 1
    []
[]

[Variables]
    [T]
        initial_condition = 300.0
    []
[]

[Kernels]
    [heat_conduction]
        type = HeatConduction
        variable = T
    []
    [time_derivative]
        type = HeatConductionTimeDerivative
        variable = T
    []
    [heat_source]
        type = HeatSource
        variable = T
        value = 1e4
    []
[]

[Materials]
    [thermal]
        type = HeatConductionMaterial
        thermal_conductivity = 45.0
        specific_heat = 0.5
    []
    [density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = 8000.0
    []
[]

[BCs]
    [t_left]
        type = DirichletBC
        variable = T
        value = 300
        boundary = 'left'
    []
    [t_right]
        type = FunctionDirichletBC
        variable = T
        function = '300+5*t'
        boundary = 'right'
    []
[]

[Executioner]
    type = Transient
    end_time = 5
    dt = 1
[]

[Outputs]
    exodus = true
[]