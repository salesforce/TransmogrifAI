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

import com.salesforce.op.utils.spark.RichVector._
import org.apache.spark.ml.linalg._

/**
 * Vector representation
 *
 * @param value vector ([[SparseVector]] or [[DenseVector]])
 */
class OPVector(val value: Vector) extends OPCollection {
  type Value = Vector

  final def isEmpty: Boolean = value.size == 0

  /**
   * Add vectors
   *
   * @param that another vector
   * @throws IllegalArgumentException if the vectors have different sizes
   * @return vector addition
   */
  def +(that: OPVector): OPVector = (value + that.value).toOPVector

  /**
   * Subtract vectors
   *
   * @param that another vector
   * @throws IllegalArgumentException if the vectors have different sizes
   * @return vector subtraction
   */
  def -(that: OPVector): OPVector = (value - that.value).toOPVector

  /**
   * Dot product between vectors
   *
   * @param that another vector
   * @throws IllegalArgumentException if the vectors have different sizes
   * @return dot product
   */
  def dot(that: OPVector): Double = value dot that.value

  /**
   * Combine multiple vectors into one
   *
   * @param that  another vector
   * @param other other vectors
   * @return result vector
   */
  def combine(that: OPVector, other: OPVector*): OPVector = value.combine(that.value, other.map(_.value): _*).toOPVector
}

object OPVector {
  def apply(value: Vector): OPVector = new OPVector(value)
  def empty: OPVector = FeatureTypeDefaults.OPVector
}
