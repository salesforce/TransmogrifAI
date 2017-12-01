/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.features.types._
import org.apache.spark.mllib.random._

import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Random

/**
 * Generator of data as real numbers
 *
 * @param randomValues the rng (from spark) that generates doubles with the specified distribution
 * @tparam DataType the feature type of the data generated
 */
case class RandomReal[DataType <: Real : WeakTypeTag]
(
  randomValues: RandomDataGenerator[Double],
  override val rng: Random = new Random
) extends FeatureFactoryOwner[DataType]
  with RandomData[DataType]
  with ProbabilityOfEmpty {
  self =>

  /**
   * Seeds random numbers generators with the given seed
   *
   * @param seed the seed
   * @return a new instance, with new rngs
   */
  override def reset(seed: Long): Unit = {
    randomValues.setSeed(seed)
    super.reset(seed)
  }

  /**
   * Infinite stream of values produced
   *
   * @return a stream of random data
   */
  def streamOfValues: InfiniteStream[Option[Double]] = new InfiniteStream[Option[Double]] {
    def next: Option[Double] = Option(randomValues.nextValue)
  }
}

object RandomReal {

  /**
   * Generator of real-number feature types with uniform distribution
   *
   * @param minValue lower bound of the values range
   * @param maxValue upper bound of the values range
   * @tparam DataType the type of data
   * @return a generator of reals
   */
  def uniform[DataType <: Real : WeakTypeTag](
    minValue: Double = 0.0, maxValue: Double = 1.0
  ): RandomReal[DataType] =
    RandomReal[DataType](new UniformDistribution(minValue, maxValue))

  /**
   * Generator of real-number feature types with normal distribution
   *
   * @param mean  the mean of the distribution
   * @param sigma the standard deviation of the distribution
   * @tparam DataType the type of data
   * @return a generator of reals
   */
  def normal[DataType <: Real : WeakTypeTag](
    mean: Double = 0.0, sigma: Double = 1.0
  ): RandomReal[DataType] =
    RandomReal[DataType](new NormalDistribution(mean, sigma))

  /**
   * Generator of real-number feature types with poisson distribution
   *
   * @param mean the mean of the distribution
   * @tparam DataType the type of data
   * @return a generator of reals
   */
  def poisson[DataType <: Real : WeakTypeTag](mean: Double = 0.0): RandomReal[DataType] =
    RandomReal[DataType](new PoissonGenerator(mean))

  /**
   * Generator of real-number feature types with exponential distribution
   *
   * @param mean the mean of the distribution
   * @tparam DataType the type of data
   * @return a generator of reals
   */
  def exponential[DataType <: Real : WeakTypeTag](mean: Double = 0.0): RandomReal[DataType] =
    RandomReal[DataType](new ExponentialGenerator(mean))

  /**
   * Generator of real-number feature types with gamma distribution
   *
   * @param shape the shape parameter of the distribution
   * @param scale the scale parameter of the distribution
   * @tparam DataType the type of data
   * @return a generator of reals
   */
  def gamma[DataType <: Real : WeakTypeTag](
    shape: Double = 5.0, scale: Double = 1.0
  ): RandomReal[DataType] =
    RandomReal[DataType](new GammaGenerator(shape, scale))


  /**
   * Generator of real-number feature types with log-normal distribution
   *
   * @param mean  the mean of the distribution
   * @param sigma the standard deviation of the distribution
   * @tparam DataType the type of data
   * @return a generator of reals
   */
  def logNormal[DataType <: Real : WeakTypeTag](
    mean: Double = 0.0, sigma: Double = 1.0
  ): RandomReal[DataType] =
    RandomReal[DataType](new LogNormalGenerator(mean, sigma))

  /**
   * Generator of real-number feature types with Weibull distribution
   *
   * @param alpha the alpha parameter of the distribution
   * @param beta  the beta parameter of the distribution
   * @tparam DataType the type of data
   * @return a generator of reals
   */
  def weibull[DataType <: Real : WeakTypeTag](alpha: Double = 1.0, beta: Double = 5.0):
  RandomReal[DataType] = RandomReal[DataType](new WeibullGenerator(alpha, beta))

  class UniformDistribution(min: Double, max: Double) extends RandomDataGenerator[Double] {
    private val source = new UniformGenerator
    override def nextValue: Double = source.nextValue * (max - min) + min
    override def copy: UniformDistribution = new UniformDistribution(min, max)
    override def setSeed(seed: Long): Unit = source.setSeed(seed)
  }

  class NormalDistribution(mean: Double, sigma: Double) extends RandomDataGenerator[Double] {
    private var source = new StandardNormalGenerator
    val coeff = math.sqrt(sigma)
    override def nextValue: Double = source.nextValue * coeff + mean
    override def copy(): NormalDistribution = new NormalDistribution(mean, sigma)
    override def setSeed(seed: Long): Unit = {
      source = new StandardNormalGenerator
      source.setSeed(seed)
    }
  }

}
