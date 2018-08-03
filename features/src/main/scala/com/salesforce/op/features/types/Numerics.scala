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

package com.salesforce.op.features.types

/**
 * Real value representation
 *
 * A base class for all the real Feature Types
 *
 * @param value real
 */
class Real(val value: Option[Double]) extends OPNumeric[Double] {
  def this(value: Double) = this(Option(value))
  final def toDouble: Option[Double] = value
  def toRealNN(default: Double): RealNN = new RealNN(value.getOrElse(default))
}
object Real {
  def apply(value: Option[Double]): Real = new Real(value)
  def apply(value: Double): Real = new Real(value)
  def empty: Real = FeatureTypeDefaults.Real
}

/**
 * Real non nullable value representation
 *
 * This value can only be constructed from a concrete [[Double]] value,
 * if empty value is passed the [[NonNullableEmptyException]] is thrown.
 *
 * @param value real
 */
class RealNN private[op](value: Option[Double]) extends Real(
    if (value == null || value.isEmpty) throw new NonNullableEmptyException(classOf[RealNN]) else value
  ) with NonNullable {
  def this(value: Double) = this(Option(value))
}
object RealNN {
  def apply(value: Double): RealNN = new RealNN(value)
}

/**
 * Binary value representation
 *
 * @param value binary
 */
class Binary(val value: Option[Boolean]) extends OPNumeric[Boolean] with SingleResponse {
  def this(value: Boolean) = this(Option(value))
  final def toDouble: Option[Double] = value.map(if (_) 1.0 else 0.0)
}
object Binary {
  def apply(value: Option[Boolean]): Binary = new Binary(value)
  def apply(value: Boolean): Binary = new Binary(value)
  def empty: Binary = FeatureTypeDefaults.Binary
}

/**
 * Integral value representation
 *
 * A base class for all the integral Feature Types
 *
 * @param value integral
 */
class Integral(val value: Option[Long]) extends OPNumeric[Long] {
  def this(value: Long) = this(Option(value))
  final def toDouble: Option[Double] = value.map(_.toDouble)
}
object Integral {
  def apply(value: Option[Long]): Integral = new Integral(value)
  def apply(value: Long): Integral = new Integral(value)
  def empty: Integral = FeatureTypeDefaults.Integral
}

/**
 * Percentage value representation
 *
 * @param value percentage
 */
class Percent(value: Option[Double]) extends Real(value) {
  def this(value: Double) = this(Option(value))
}
object Percent {
  def apply(value: Option[Double]): Percent = new Percent(value)
  def apply(value: Double): Percent = new Percent(value)
  def empty: Percent = FeatureTypeDefaults.Percent
}

/**
 * Currency value representation
 *
 * @param value currency
 */
class Currency(value: Option[Double]) extends Real(value) {
  def this(value: Double) = this(Option(value))
}
object Currency {
  def apply(value: Option[Double]): Currency = new Currency(value)
  def apply(value: Double): Currency = new Currency(value)
  def empty: Currency = FeatureTypeDefaults.Currency
}

/**
 * Date value representation
 *
 * @param value date (assumed to be in ms since Epoch)
 */
class Date(value: Option[Long]) extends Integral(value) {
  def this(value: Long) = this(Option(value))
}
object Date {
  def apply(value: Option[Long]): Date = new Date(value)
  def apply(value: Long): Date = new Date(value)
  def empty: Date = FeatureTypeDefaults.Date
}

/**
 * Date & time value representation
 *
 * @param value date & time (assumed to be in ms since Epoch)
 */
class DateTime(value: Option[Long]) extends Date(value) {
  def this(value: Long) = this(Option(value))
}
object DateTime {
  def apply(value: Option[Long]): DateTime = new DateTime(value)
  def apply(value: Long): DateTime = new DateTime(value)
  def empty: DateTime = FeatureTypeDefaults.DateTime
}

