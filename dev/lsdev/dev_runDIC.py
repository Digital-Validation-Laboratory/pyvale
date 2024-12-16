from pathlib import Path
import muDIC as dic
import logging

def main() -> None:
    logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s', level=logging.INFO)

    # Path to folder containing images
    path = str(Path('dev/lsdev/rendered_images/RBM_1-5mm/')) # Use this formatting on Linux and Mac OS

    # Generate image instance containing all images found in the folder
    images = dic.IO.image_stack_from_folder(path, file_type='.tiff')
    #images.set_filter(dic.filtering.lowpass_gaussian, sigma=1.)


    # Generate mesh
    mesher = dic.Mesher(deg_e=3, deg_n=3,type="q4")

    # If you want to see use a GUI, set GUI=True below
    mesh = mesher.mesh(images,n_ely=20,n_elx=20, GUI=True)

    # Instantiate settings object and set some settings manually
    settings = dic.DICInput(mesh, images)
    settings.max_nr_im = 500
    settings.maxit = 20
    settings.tol = 1.e-6
    settings.interpolation_order = 4
    # If you want to access the residual fields after the analysis, this should be set to True
    settings.store_internals = True

    # This setting defines the behaviour when convergence is not obtained
    settings.noconvergence = "ignore"

    # Instantiate job object
    job = dic.DICAnalysis(settings)

    # Running DIC analysis
    dic_results = job.run()

    # Calculate field values
    fields = dic.post.viz.Fields(dic_results,upscale=1)

    # Show a field
    viz = dic.Visualizer(fields,images=images)
    viz.show(field="displacement", component = (1,1), frame=0)


    # # Get images
    # path = str(Path('dev/lsdev/rendered_images/RBM_1-5mm/'))
    # image_stack = dic.image_stack_from_folder(path, file_type='.tiff')

    # #Mesh images
    # mesher = dic.Mesher()

    # mesh = mesher.mesh(image_stack, n_elx=20, n_ely=20)
    # print(f"{mesh.element_def=}")


    # # Solver settings
    # settings = dic.DICInput(image_stack, mesh)
    # settings.interpolation_order = 4
    # settings.noconvergence = 'ignore'

    # # Running DIC solver
    # job = dic.DICAnalysis(settings)
    # dic_results = job.run()

    # # Calculate field values
    # fields = dic.post.viz.Fields(dic_results, upscale=10)

    # viz = dic.Visualizer(fields, images=image_stack)

    # viz.show(field='displacement', component=(1,1), frame=1)

if __name__ == '__main__':
    main()