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

package com.salesforce.op.stages.impl.selector

import com.salesforce.op.stages.impl.selector.Distribution.{Exponential, Subset, Uniform}
import enumeratum._
import org.apache.spark.ml.param._

import scala.util.Random
import scala.collection.mutable

sealed abstract class Distribution extends EnumEntry with Serializable

private object Distribution extends Enum[Distribution] {
  val values: Seq[Distribution] = findValues
  case object Uniform extends Distribution
  case object Subset extends Distribution
  case object Exponential extends Distribution
}

/**
 * Builder for a param sets used in random search-based model selection.
 */
class RandomParamBuilder(random: Random = new Random()) {

  private val paramDefs = mutable.Map.empty[Param[_], (Distribution, _, _, Seq[_])]

  private def addParams[T](param: Param[T], dist: Distribution, min: T, max: T): this.type = {
    paramDefs.put(param, (dist, min, max, Seq()))
    this
  }


  /**
   * Adds parameter values uniformly selected from a sequence
   *
   * @param param parameter to add
   *
   * @param seq   sequence of possible values
   */
  def subset[T](param: Param[T], seq: Seq[T]): this.type = {
    paramDefs.put(param, (Subset, None, None, seq))
    this
  }

  /**
   * Adds double param with range of values in uniform distribution
   *
   * @param param parameter to add
   * @param min   minimum value for param
   * @param max   maximum value for param
   */
  def uniform(param: DoubleParam, min: Double, max: Double): this.type = {
    require(min < max, "min must be less than max")
    addParams[Double](param, Uniform, min, max)
  }

  /**
   * Adds float param with range of values in uniform distribution
   *
   * @param param parameter to add
   * @param min   minimum value for param
   * @param max   maximum value for param
   */
  def uniform(param: FloatParam, min: Float, max: Float): this.type = {
    require(min < max, "min must be less than max")
    addParams[Float](param, Uniform, min, max)
  }

  /**
   * Adds int param with range of values in uniform distribution
   *
   * @param param parameter to add
   * @param min   minimum value for param
   * @param max   maximum value for param
   */
  def uniform(param: IntParam, min: Int, max: Int): this.type = {
    require(min < max, "min must be less than max")
    addParams[Int](param, Uniform, min, max)
  }

  /**
   * Adds long param with range of values in uniform distribution
   *
   * @param param parameter to add
   * @param min   minimum value for param
   * @param max   maximum value for param
   */
  def uniform(param: LongParam, min: Long, max: Long): this.type = {
    require(min < max, "min must be less than max")
    addParams[Long](param, Uniform, min, max)
  }

  /**
   * Adds boolean param
   *
   * @param param parameter to add
   */
  def uniform(param: BooleanParam): this.type = {
    addParams[Boolean](param, Uniform, false, true)
  }

  /**
   * Adds double param with range of values exponentially distributed in range specified (must be in (0, +Inf)
   *
   * @param param parameter to add
   * @param min   minimum value for param
   * @param max   maximum value for param
   */
  def exponential(param: DoubleParam, min: Double, max: Double): this.type = {
    require(min > 0, "Min value must be greater than zero for exponential distribution to work")
    require(min < max, "min must be less than max")
    addParams[Double](param, Exponential, min, max)
  }

  /**
   * Adds float param with range of values exponentially distributed in range specified (must be in (0, +Inf)
   *
   * @param param parameter to add
   * @param min   minimum value for param
   * @param max   maximum value for param
   */
  def exponential(param: FloatParam, min: Float, max: Float): this.type = {
    require(min > 0, "Min value must be greater than zero for exponential distribution to work")
    require(min < max, "min must be less than max")
    addParams[Float](param, Exponential, min, max)
  }

  private def getExpRand(min: Double, max: Double) = {
    val minExp = math.log10(min)
    val maxExp = math.log10(max)
    val exp = (maxExp - minExp) * random.nextDouble() + minExp
    math.pow(10, exp)
  }

  /**
   * Builds a set of parameters to try for the specified params
   * @param totalParams number of parameter settings to try
   * @return An array of param maps containing randomly generated values for each param specified
   */
  def build(totalParams: Int): Array[ParamMap] = {
    val allParams = for { _ <- 0 until totalParams } yield {
      val params = new ParamMap()
      paramDefs.foreach {
        case (param, (Subset, _, _, seq: Seq[_])) =>
          params.put(param.asInstanceOf[Param[Any]], seq(random.nextInt(seq.length)))
        case (param, (Uniform, min: Double, max: Double, _)) =>
          params.put(param.asInstanceOf[Param[Any]], (max - min) * random.nextDouble() + min)
        case (param, (Uniform, min: Float, max: Float, _)) =>
          params.put(param.asInstanceOf[Param[Any]], (max - min) * random.nextFloat() + min)
        case (param, (Uniform, min: Int, max: Int, _)) =>
          params.put(param.asInstanceOf[Param[Any]], random.nextInt(max - min) + min)
        case (param, (Uniform, min: Long, max: Long, _)) =>
          val range = math.min(max - min, Integer.MAX_VALUE.toLong).toInt
          params.put(param.asInstanceOf[Param[Any]], random.nextInt(range).toLong + min)
        case (param, (Uniform, _: Boolean, _: Boolean, _)) =>
          params.put(param.asInstanceOf[Param[Any]], random.nextBoolean())
        case (param, (Exponential, min: Double, max: Double, _)) =>
          params.put(param.asInstanceOf[Param[Any]], getExpRand(min, max))
        case (param, (Exponential, min: Float, max: Float, _)) =>
          params.put(param.asInstanceOf[Param[Any]], getExpRand(min, max))
        case (_, _) => throw new IllegalArgumentException("parameter type and distribution not supported")
      }
      params
    }
    allParams.toArray
  }
}
