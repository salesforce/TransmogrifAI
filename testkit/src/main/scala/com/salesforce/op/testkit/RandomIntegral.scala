/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import java.util.{Date => JDate}

import com.salesforce.op.features.types._

import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Random

/**
 * Generator of data as integral numbers
 *
 * @param numbers            the stream of longs used as the source
 * @tparam DataType the feature type of the data generated
 */
case class RandomIntegral[DataType <: Integral : WeakTypeTag]
(
  numbers: RandomStream[Long]
) extends StandardRandomData[DataType](
  numbers map (Option(_))
) with ProbabilityOfEmpty

/**
 * Generator of data as integral numbers
 */
object RandomIntegral {

  /**
   * Generator of random integral values
   *
   * @return generator of integrals
   */
  def integrals: RandomIntegral[Integral] =
    RandomIntegral[Integral](RandomStream.ofLongs)

  /**
   * Generator of random integral values in a given range
   *
   * @param from minimum value to produce (inclusive)
   * @param to   maximum value to produce (exclusive)
   * @return the generator of integrals
   */
  def integrals(from: Long, to: Long): RandomIntegral[Integral] =
    RandomIntegral[Integral](RandomStream.ofLongs(from, to))

  /**
   * Incremental generator of dates.
   * Dates are generated incrementally, with the step in millis that is
   * between minStep and maxStep.
   *
   * @param init    initial date starting from which to generate
   * @param minStep next date will be at least this many millis after the previous
   * @param maxStep next date will be at most this many millis after the previous (-1 if constant
   * @return the generator of dates
   */
  def dates(init: JDate, minStep: Int, maxStep: Int): RandomIntegral[Date] = {
    RandomIntegral[Date](incrementingStream(init, minStep, maxStep))
  }

  /**
   * Incremental generator of datetimes
   *
   * @param init    initial date starting from which to generate
   * @param minStep next datetime will be at least this many millis after the previous
   * @param maxStep next datetime will be at most this many millis after the previous
   * @return the generator of datetimes
   */
  def datetimes(init: JDate, minStep: Int, maxStep: Int): RandomIntegral[DateTime] = {
    RandomIntegral[DateTime](incrementingStream(init, minStep, maxStep))
  }

  private def incrementingStream(init: JDate, minStep: Long, maxStep: Long): RandomStream[Long] =
    RandomStream.incrementing(init.getTime, minStep, maxStep)
}


