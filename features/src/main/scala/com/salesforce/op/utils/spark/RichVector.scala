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

package com.salesforce.op.utils.spark

import breeze.linalg.{DenseVector => BreezeDenseVector, SparseVector => BreezeSparseVector, Vector => BreezeVector}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector}

/**
 * [[org.apache.spark.ml.linalg.Vector]] enrichment functions
 */
object RichVector {

  implicit class RichVector(val v: Vector) extends AnyVal {

    /**
     * Add vectors
     *
     * @param that another vector
     * @throws IllegalArgumentException if the vectors have different sizes
     * @return vector addition
     */
    def +(that: Vector): Vector = {
      val res = v.toBreeze + that.toBreeze
      toSpark(res)
    }

    /**
     * Subtract vectors
     *
     * @param that another vector
     * @throws IllegalArgumentException if the vectors have different sizes
     * @return vector subtraction
     */
    def -(that: Vector): Vector = {
      val res = v.toBreeze - that.toBreeze
      toSpark(res)
    }

    /**
     * Convert to [[breeze.linalg.Vector]]
     *
     * @return [[breeze.linalg.Vector]]
     */
    def toBreeze: BreezeVector[Double] = v match {
      case s: SparseVector => new BreezeSparseVector[Double](s.indices, s.values, s.size)
      case d: DenseVector => new BreezeDenseVector[Double](d.values)
    }

    /**
     * Convert [[breeze.linalg.Vector]] back to [[org.apache.spark.ml.linalg.Vector]]
     * @return [[org.apache.spark.ml.linalg.Vector]]
     */
    private def toSpark: BreezeVector[Double] => Vector = {
      case s: BreezeSparseVector[Double]@unchecked => new SparseVector(s.length, s.index, s.data)
      case d: BreezeDenseVector[Double]@unchecked => new DenseVector(d.data)
    }

  }

}
