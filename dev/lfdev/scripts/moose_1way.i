# substituting a for you appropiate variable, and the mesh variable in the uo

[AuxVariables]
    [a]
      family = LAGRANGE
      order = FIRST
    []
  []

  [AuxKernels]
    [a_k]
      type = SolutionAux
      variable = a
      solution = solution_uo
      from_variable = a
      execute_on = 'initial timestep_begin'
    []
  []

  [UserObjects]
    [solution_uo]
      type = SolutionUserObject
      mesh = build_out.e
    []
  []