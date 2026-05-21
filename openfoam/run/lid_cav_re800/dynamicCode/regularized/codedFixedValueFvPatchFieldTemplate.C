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
    // SHA1 = 284c828b5f77f40b42a2726664bde4dd77b4d94b
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void regularized_284c828b5f77f40b42a2726664bde4dd77b4d94b(bool load)
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
    regularizedFixedValueFvPatchVectorField
);


const char* const regularizedFixedValueFvPatchVectorField::SHA1sum =
    "284c828b5f77f40b42a2726664bde4dd77b4d94b";


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

regularizedFixedValueFvPatchVectorField::
regularizedFixedValueFvPatchVectorField
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
        Info<<"construct regularized sha1: 284c828b5f77f40b42a2726664bde4dd77b4d94b"
            " from patch/dictionary\n";
    }
}


regularizedFixedValueFvPatchVectorField::
regularizedFixedValueFvPatchVectorField
(
    const regularizedFixedValueFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper)
{
    if (false)
    {
        Info<<"construct regularized sha1: 284c828b5f77f40b42a2726664bde4dd77b4d94b"
            " from patch/DimensionedField/mapper\n";
    }
}


regularizedFixedValueFvPatchVectorField::
regularizedFixedValueFvPatchVectorField
(
    const regularizedFixedValueFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF)
{
    if (false)
    {
        Info<<"construct regularized sha1: 284c828b5f77f40b42a2726664bde4dd77b4d94b "
            "as copy/DimensionedField\n";
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

regularizedFixedValueFvPatchVectorField::
~regularizedFixedValueFvPatchVectorField()
{
    if (false)
    {
        Info<<"destroy regularized sha1: 284c828b5f77f40b42a2726664bde4dd77b4d94b\n";
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void regularizedFixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        Info<<"updateCoeffs regularized sha1: 284c828b5f77f40b42a2726664bde4dd77b4d94b\n";
    }

//{{{ begin code
    #line 26 "/home/openfoam/run/lid_cav_re800/0/U/boundaryField/movingWall"
const vectorField& Cf = patch().Cf();
		vectorField& field = *this;
		forAll(Cf, faceI)
		{
			const scalar x = Cf[faceI][0];
			field[faceI] = vector((1-x)*(1-x)*(1+x)*(1+x),0,0);
		}
//}}} end code

    this->fixedValueFvPatchField<vector>::updateCoeffs();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

