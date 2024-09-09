run_vtk = False
if run_vtk:
    import vtk
    vtk_mathtext = vtk.vtkMathTextFreeTypeTextRenderer()
    print(vtk_mathtext.MathTextIsSupported())

import pyvista as pv
plotter = pv.Plotter()
plotter.add_text(r'$\rho$')
plotter.show()
