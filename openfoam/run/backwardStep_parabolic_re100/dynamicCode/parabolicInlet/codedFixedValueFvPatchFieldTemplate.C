/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) YEAR OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "codedFixedValueFvPatchFieldTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
//{{{ begin codeInclude

//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    // dynamicCode:
    // SHA1 = fec44b96033fd1a268374e18b7036ced509664f4
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void parabolicInlet_fec44b96033fd1a268374e18b7036ced509664f4(bool load)
    {
        if (load)
        {
            // code that can be explicitly executed after loading
        }
        else
        {
            // code that can be explicitly executed before unloading
        }
    }
}

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

makeRemovablePatchTypeField
(
    fvPatchVectorField,
    parabolicInletFixedValueFvPatchVectorField
);


const char* const parabolicInletFixedValueFvPatchVectorField::SHA1sum =
    "fec44b96033fd1a268374e18b7036ced509664f4";


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

parabolicInletFixedValueFvPatchVectorField::
parabolicInletFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict)
{
    if (false)
    {
        Info<<"construct parabolicInlet sha1: fec44b96033fd1a268374e18b7036ced509664f4"
            " from patch/dictionary\n";
    }
}


parabolicInletFixedValueFvPatchVectorField::
parabolicInletFixedValueFvPatchVectorField
(
    const parabolicInletFixedValueFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper)
{
    if (false)
    {
        Info<<"construct parabolicInlet sha1: fec44b96033fd1a268374e18b7036ced509664f4"
            " from patch/DimensionedField/mapper\n";
    }
}


parabolicInletFixedValueFvPatchVectorField::
parabolicInletFixedValueFvPatchVectorField
(
    const parabolicInletFixedValueFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF)
{
    if (false)
    {
        Info<<"construct parabolicInlet sha1: fec44b96033fd1a268374e18b7036ced509664f4 "
            "as copy/DimensionedField\n";
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

parabolicInletFixedValueFvPatchVectorField::
~parabolicInletFixedValueFvPatchVectorField()
{
    if (false)
    {
        Info<<"destroy parabolicInlet sha1: fec44b96033fd1a268374e18b7036ced509664f4\n";
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void parabolicInletFixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        Info<<"updateCoeffs parabolicInlet sha1: fec44b96033fd1a268374e18b7036ced509664f4\n";
    }

//{{{ begin code
    #line 27 "/run/backwardStep_parabolic_re200/0/U/boundaryField/inlet"
const vectorField& Cf = patch().Cf();
        vectorField& U = *this;

        const scalar y0   = 0.1;   // bottom of inlet (meters)
        const scalar H    = 0.1;   // inlet height (meters) = 1.0*convertToMeters
        const scalar Ubar = 1.0;   // desired MEAN inlet velocity

        forAll(Cf, faceI)
        {
            scalar y = Cf[faceI].y() - y0;                // y in [0,H]
            scalar Ux = 6.0*Ubar*(y*(H - y))/(H*H);       // mean(Ux)=Ubar
            U[faceI] = vector(Ux, 0, 0);
        }
//}}} end code

    this->fixedValueFvPatchField<vector>::updateCoeffs();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

