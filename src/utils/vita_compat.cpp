//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2020 SuperTuxKart-Team
//
//  This program is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public License
//  as published by the Free Software Foundation; either version 3
//  of the License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

#ifdef VITA

#include <math.h>

/** Small compatibility shims for the PS Vita toolchain.
 *
 *  GCC 15 recognises round-half-to-even idioms in lib/graphics_utils and
 *  lowers them to calls to the C23 roundeven family, but the newlib shipped
 *  with vitasdk does not provide them. rintf/rint are exactly round-half-to-
 *  even while the default rounding mode (FE_TONEAREST) is in effect, which
 *  STK never changes, so they are a faithful implementation here.
 */
extern "C" float roundevenf(float x)
{
    return rintf(x);
}   // roundevenf

// ----------------------------------------------------------------------------
extern "C" double roundeven(double x)
{
    return rint(x);
}   // roundeven

#endif   // VITA
