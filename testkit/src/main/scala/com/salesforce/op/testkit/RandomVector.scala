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

package com.salesforce.op.testkit

import language.postfixOps
import com.salesforce.op.features.types._
import org.apache.spark.ml.linalg._

import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Random
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW

/**
 * Generator of vectors
 *
 * @param values the stream of longs used as the source
 */
case class RandomVector(values: RandomStream[Vector])
  extends StandardRandomData[OPVector](values)

object RandomVector {

  private def asDense(size: Int, source: Iterator[Double]) =
    new DenseVector(source take size toArray)

  private def asSparse(size: Int, source: Iterator[Option[Double]]): SparseVector = {
    val indexed = source.take(size).zipWithIndex collect { case (Some(x), i) => i -> x } toArray

    new SparseVector(size, indexed map (_._1), indexed map (_._2))
  }

  /**
   * Builds a sparse vector out of a given partial function; a Map is also a partial function
   * @param size vector size
   * @param source function that produces (or not) a value at a given index
   * @return
   */
  private def asSparse(size: Int, source: PartialFunction[Int, Double]): SparseVector = {
    asSparse(size, 0 to size map source.lift iterator)
  }

  type RandomReals = RandomData[Real]
  type RandomVectors = RandomStream[Vector]

  /**
   * Generating dense vectors of given length, from a real value generator.
   * There is a big reason why RandomDataGenerator is not involved here. A bug in spark.
   * @param length vector length
   * @param valueGenerator generator of individual values
   * @return a stream of dense vectors
   */
  private def denseVectors(length: Int, valueGenerator: RandomReals): RandomVectors = {
    val iteratorOfOptions = valueGenerator.streamOfValues
    val iteratorOfDoubles: Iterator[Double] = iteratorOfOptions collect { case Some(x) => x }

    new RandomVectors(_ => asDense(length, iteratorOfDoubles)) {
      override def reset(seed: Long): Unit = {
        valueGenerator.reset(seed)
      }
    }
  }

  private def denseVectorsFromStream(length: Int, values: RandomStream[Double]): RandomVectors = {
    new RandomVectors(rng => asDense(length, values(rng)))
  }

  /**
   * Generating dense vectors of given length, from a real value generator.
   * There is a big reason why RandomDataGenerator is not involved here. A bug in spark.
   * @param length vector length
   * @param valueGenerator generator of individual values
   * @return
   */
  private def sparseVectors(length: Int, valueGenerator: RandomReals): RandomVectors = {
    val iteratorOfReals: Iterator[Real] = valueGenerator
    val iteratorOfValues: Iterator[Option[Double]] = iteratorOfReals map (_.value)

    new RandomVectors(_ => asSparse(length, iteratorOfValues)) {
      override def reset(seed: Long): Unit = { valueGenerator.reset(seed) }
    }
  }

  private def sparseVectors(length: Int, values: RandomStream[Option[Double]]): RandomVectors = {
    new RandomVectors(rng => asSparse(length, values(rng)))
  }

  /**
   * Produces random dense vectors with a given distribution
   *
   * @param dataSource  generator of random reals to be stored in the generated vectors
   * @param length vectors length (they are all of the same length)
   * @return a generator of sets of texts
   */
  def dense(dataSource: RandomReals, length: Int): RandomVector = {
    RandomVector(denseVectors(length, dataSource))
  }

  /**
   * Produces random sparse vectors with a given distribution
   *
   * @param dataSource  generator of random reals to be stored in the generated vectors
   * @param length vectors length (they are all of the same length)
   * @return a generator of sets of texts
   */
  def sparse(dataSource: RandomReals, length: Int): RandomVector = {
    RandomVector(sparseVectors(length, dataSource))
  }

  private def add(xs: Array[Double], ys: Array[Double]): Array[Double] = {
    require(xs.length == ys.length,
      s"Expected lengths to be the same, got ${xs.length} and ${ys.length}")
    val zs = new Array[Double](xs.length)

    for { i <- xs.indices } zs(i) = xs(i) + ys(i)
    zs
  }

  private def add(xs: Vector, ys: Vector): Vector = {
    new DenseVector(add(xs.toDense.values, ys.toDense.values))
  }

  private def cholesky(m: Matrix): Matrix = {
    val n = m.numCols
    val resultCode = new intW(0)
    val a = m.toArray
    for {
      i <- 0 until n
      j <- i + 1 until n
    } {
      a(i + j * n) = 0.0
    }

    lapack.dpotrf("L", n, a, n, resultCode)

    resultCode.`val` match {
      case code if code < 0 =>
        throw new IllegalStateException(s"LAPACK returned $code; arg ${-code} is illegal")
      case code if code > 0 =>
        throw new IllegalArgumentException (
          s"LAPACKreturned $code because matrix is not positive definite.")
      case _ => // do nothing
    }

    new DenseMatrix(n, n, a)
  }

  /**
   * Produces normally distributed random vectors with a given mean and covariance matrix.
   * @param mean the mean value of generated vectors
   * @param covMatrix the covariance matrix of generate vectors
   * @return a RandomVector generator that produces vectors satisfying the given conditions
   */
  def normal(mean: Vector, covMatrix: Matrix): RandomVector = {
    require(covMatrix.numCols == covMatrix.numRows,
      s"Expected square covariance matrix, got ${covMatrix.numRows}x${covMatrix.numCols}")
    require(mean.size == covMatrix.numRows,
      s"Expected mean vector size ${covMatrix.numRows}, got ${mean.size}")
    val transform = cholesky(covMatrix)
    val source = denseVectors(mean.size, RandomReal.normal())
    val vectors = source map (v => add(mean, transform.multiply(v)))
    new RandomVector(vectors)
  }

  /**
   * Produces random OPVectors (with DenseVector inside) consisting of zeroes and ones.
   * The probability of 1 is given.
   *
   * @param size dimension of the vectors
   * @param probabilityOfOne probability at which we have 1.
   * @return a generator of random vectors.
   */
  def binary(size: Int, probabilityOfOne: Double): RandomVector = {
    val rnb = RandomStream.ofOnesAndZeros(probabilityOfOne)
    RandomVector(denseVectorsFromStream(size, rnb))
  }

  /**
   * Produces random OPVectors (with DenseVector inside) consisting of zeroes and ones.
   * For each component we are given the probability of 1.
   *
   * @param probabilitiesOfOne probability at which we have 1, specified per component.
   * @return a generator of random vectors.
   */
  def binary(probabilitiesOfOne: Seq[Double]): RandomVector = {
    val size = probabilitiesOfOne.size
    val streams = probabilitiesOfOne map RandomStream.ofOnesAndZeros
    def values(rng: Random) = streams.map(_.apply(rng).next)

    def vectors(rng: Random) = asDense(size, values(rng).iterator)

    RandomVector(new RandomVectors(vectors))
  }

}
