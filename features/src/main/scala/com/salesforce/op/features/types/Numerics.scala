/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.features.types


class Real(val value: Option[Double]) extends OPNumeric[Double] {
  def this(value: Double) = this(Option(value))
  final def toDouble: Option[Double] = value
  def toRealNN(default: Option[Double] = FeatureTypeDefaults.RealNN.value): RealNN =
    new RealNN(value.orElse(default))
}
object Real {
  def apply(value: Option[Double]): Real = new Real(value)
  def apply(value: Double): Real = new Real(value)
  def empty: Real = FeatureTypeDefaults.Real
}

class RealNN(value: Option[Double]) extends Real(value.orElse(FeatureTypeDefaults.RealNN.value)) with NonNullable {
  def this(value: Double) = this(Option(value))
}
object RealNN {
  def apply(value: Option[Double]): RealNN = new RealNN(value.orElse(FeatureTypeDefaults.RealNN.value))
  def apply(value: Double): RealNN = new RealNN(value)
  def empty: RealNN = FeatureTypeDefaults.RealNN
}

class Binary(val value: Option[Boolean]) extends OPNumeric[Boolean] with SingleResponse {
  def this(value: Boolean) = this(Option(value))
  final def toDouble: Option[Double] = value.map { case true => 1.0; case false => 0.0 }
}
object Binary {
  def apply(value: Option[Boolean]): Binary = new Binary(value)
  def apply(value: Boolean): Binary = new Binary(value)
  def empty: Binary = FeatureTypeDefaults.Binary
}

class Integral(val value: Option[Long]) extends OPNumeric[Long] {
  def this(value: Long) = this(Option(value))
  final def toDouble: Option[Double] = value.map(_.toDouble)
}
object Integral {
  def apply(value: Option[Long]): Integral = new Integral(value)
  def apply(value: Long): Integral = new Integral(value)
  def empty: Integral = FeatureTypeDefaults.Integral
}

class Percent(value: Option[Double]) extends Real(value) {
  def this(value: Double) = this(Option(value))
}
object Percent {
  def apply(value: Option[Double]): Percent = new Percent(value)
  def apply(value: Double): Percent = new Percent(value)
  def empty: Percent = FeatureTypeDefaults.Percent
}

class Currency(value: Option[Double]) extends Real(value) {
  def this(value: Double) = this(Option(value))
}
object Currency {
  def apply(value: Option[Double]): Currency = new Currency(value)
  def apply(value: Double): Currency = new Currency(value)
  def empty: Currency = FeatureTypeDefaults.Currency
}

class Date(value: Option[Long]) extends Integral(value) {
  def this(value: Long) = this(Option(value))
}
object Date {
  def apply(value: Option[Long]): Date = new Date(value)
  def apply(value: Long): Date = new Date(value)
  def empty: Date = FeatureTypeDefaults.Date
}

class DateTime(value: Option[Long]) extends Date(value) {
  def this(value: Long) = this(Option(value))
}
object DateTime {
  def apply(value: Option[Long]): DateTime = new DateTime(value)
  def apply(value: Long): DateTime = new DateTime(value)
  def empty: DateTime = FeatureTypeDefaults.DateTime
}

