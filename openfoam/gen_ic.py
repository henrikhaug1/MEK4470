import numpy as np

N = 64
L = 2 * np.pi
dx = L / N

x = np.linspace(dx/2, L - dx/2, N)
y = np.linspace(dx/2, L - dx/2, N)
X, Y = np.meshgrid(x, y, indexing="ij")

Ux = np.sin(X) * np.cos(Y)
Uy = -np.cos(X) * np.sin(Y)
p  = (np.cos(2*X) + np.cos(2*Y)) / 4.0

boundary = """
boundaryField
{
    left        { type cyclic; }
    right       { type cyclic; }
    top         { type cyclic; }
    bottom      { type cyclic; }
    frontAndBack { type empty; }
}
"""

path = "openfoam/run/taylor_green_re200/0/"

with open(path + "p", "w", buffering=1) as f:
    f.write('FoamFile\n{\n    format      ascii;\n    class       volScalarField;\n    object      p;\n}\n')
    f.write('dimensions      [0 2 -2 0 0 0 0];\n')
    f.write('internalField   nonuniform List<scalar>\n')
    f.write(f'{N*N}\n(\n')
    for j in range(N):
        for i in range(N):
            f.write(f'{p[i,j]:.10f}\n')
    f.write(')\n;\n')
    f.write(boundary)
    f.flush()

with open(path + "U", "w", buffering=1) as f:
    f.write('FoamFile\n{\n    format      ascii;\n    class       volVectorField;\n    object      U;\n}\n')
    f.write('dimensions      [0 1 -1 0 0 0 0];\n')
    f.write('internalField   nonuniform List<vector>\n')
    f.write(f'{N*N}\n(\n')
    for j in range(N):
        for i in range(N):
            f.write(f'({Ux[i,j]:.10f} {Uy[i,j]:.10f} 0)\n')
    f.write(')\n;\n')
    f.write(boundary)
    f.flush()

print("Done")
