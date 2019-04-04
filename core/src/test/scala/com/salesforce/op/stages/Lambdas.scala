package com.salesforce.op.stages

import com.salesforce.op.features.types.Real
import com.salesforce.op.features.types._

/**
 * @author ksuchanek
 * @since 214
 */
object Lambdas {
  def fncUnary = (x: Real) => x.v.map(_ * 0.1234).toReal

  def fncSequence = (x: Seq[DateList]) => {
    val v = x.foldLeft(0.0)((a, b) => a + b.value.sum)
    Math.round(v / 1E6).toReal
  }

  def fncBinarySequence = (y: Real, x: Seq[DateList]) => {
    val v = x.foldLeft(0.0)((a, b) => a + b.value.sum)
    (Math.round(v / 1E6) + y.value.getOrElse(0.0)).toReal
  }


  def fncBinary: (Real, Real) => Real = (x: Real, y: Real) => y.v.flatMap((yv: Double) => x.value.map(_ * yv)).toReal

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
