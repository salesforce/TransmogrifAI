/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.testkit

import com.salesforce.op.features.types._
import com.salesforce.op.testkit.RandomSet.setsOf

import scala.reflect.runtime.universe.WeakTypeTag
import scala.util.Random

/**
 * Generator of sets
 *
 * @param values the stream of longs used as the source
 * @tparam DataType the type of data in sets
 * @tparam SetType  the feature type of the data generated
 */
case class RandomSet[DataType, SetType <: OPSet[DataType] : WeakTypeTag]
(
  values: RandomStream[Set[DataType]]
) extends StandardRandomData[SetType](
  sourceOfData = values.map(_.asInstanceOf[SetType#Value])
)

object RandomMultiPickList {

  /**
   * Produces random sets of MultiPickList
   *
   * @param texts  generator of random texts to be stored in the generated sets
   * @param minLen minimum length of the set; 0 if missing
   * @param maxLen maximum length of the set; if missing, all sets are of the same size
   * @return a generator of sets of texts
   */
  def of(texts: RandomText[_], minLen: Int = 0, maxLen: Int = -1): RandomSet[String, MultiPickList] = {
    setsOf[String, MultiPickList](texts.stream, minLen, maxLen)
  }

}

object RandomSet {

  /**
   * Produces random sets of texts
   *
   * @param texts  generator of random texts to be stored in the generated sets
   * @param minLen minimum length of the set; 0 if missing
   * @param maxLen maximum length of the set; if missing, all sets are of the same size
   * @return a generator of sets of texts
   */
  def of(texts: RandomText[_], minLen: Int = 0, maxLen: Int = -1): RandomSet[String, OPSet[String]] =
    setsOf[String, OPSet[String]](texts.stream, minLen, maxLen)

  private[testkit] def setsOf[D, T <: OPSet[D] : WeakTypeTag](stream: RandomStream[D], minLen: Int, maxLen: Int) =
    RandomSet[D, T](RandomStream.groupInChunks[D](minLen, maxLen)(stream) map (_.toSet))
}
