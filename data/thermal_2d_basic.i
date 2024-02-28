#
# Single block thermal input with boundary conditions
# https://mooseframework.inl.gov/modules/heat_transfer/tutorials/introduction/therm_step02.html
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
    []
[]

[Kernels]
    [heat_conduction]
        type = HeatConduction
        variable = T
    []
[]

[Materials]
    [thermal]
        type = HeatConductionMaterial
        thermal_conductivity = 45.0
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