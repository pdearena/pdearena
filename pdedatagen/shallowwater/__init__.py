from subprocess import run
import os.path

def generate_trajectories_shallowwater(savedir, n_samples, seed):
    import juliapkg
    juliapkg.resolve()
    file = os.path.join(os.path.dirname(__file__), 'datagen.jl')
    run([juliapkg.executable(), f'--project={juliapkg.project()}', '--startup-file=no', file, savedir, n_samples, seed])