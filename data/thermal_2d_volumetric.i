#
# Single block thermal input with time derivative and volumetric heat source terms
# https://mooseframework.inl.gov/modules/heat_transfer/tutorials/introduction/therm_step03.html
#

init_temp = 0.0

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
        initial_condition = ${init_temp}
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
        value = ${init_temp}
        boundary = 'left'
    []
    [t_right]
        type = FunctionDirichletBC
        variable = T
        function = '${init_temp} + 10*t'
        boundary = 'right'
    []
[]

[Executioner]
    type = Transient
    end_time = 20
    dt = 1
[]

[Outputs]
    exodus = true
[]