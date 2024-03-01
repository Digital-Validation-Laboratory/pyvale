import vtk
vtk_mathtext = vtk.vtkMathTextFreeTypeTextRenderer()
print(vtk_mathtext.MathTextIsSupported())

import vtk

actor = vtk.vtkTextActor()
actor.SetInput(r'$\rho$')
actor.GetTextProperty().SetFontSize(75)
ren = vtk.vtkRenderer()
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
ren.AddActor(actor)
iren.Initialize()
renWin.Render()
iren.Start()