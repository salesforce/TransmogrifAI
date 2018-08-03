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
