/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages

import com.salesforce.op.features.types.Real
import com.salesforce.op.features.types._

object Lambdas {
  def fncUnary: Real => Real = (x: Real) => x.v.map(_ * 0.1234).toReal

  def fncSequence: Seq[DateList] => Real = (x: Seq[DateList]) => {
    val v = x.foldLeft(0.0)((a, b) => a + b.value.sum)
    Math.round(v / 1E6).toReal
  }

  def fncBinarySequence: (Real, Seq[DateList]) => Real = (y: Real, x: Seq[DateList]) => {
    val v = x.foldLeft(0.0)((a, b) => a + b.value.sum)
    (Math.round(v / 1E6) + y.value.getOrElse(0.0)).toReal
  }

  def fncBinary: (Real, Real) => Real = (x: Real, y: Real) => (
    for {
      yv <- y.value
      xv <- x.value
    } yield xv * yv
    ).toReal

  def fncTernary: (Real, Real, Real) => Real = (x: Real, y: Real, z: Real) =>
    (for {
      xv <- x.value
      yv <- y.value
      zv <- z.value
    } yield xv * yv + zv).toReal

  def fncQuaternary: (Real, Real, Text, Real) => Real = (x: Real, y: Real, t: Text, z: Real) =>
    (for {
      xv <- x.value
      yv <- y.value
      tv <- t.value
      zv <- z.value
    } yield xv * yv + zv * tv.length).toReal

}
