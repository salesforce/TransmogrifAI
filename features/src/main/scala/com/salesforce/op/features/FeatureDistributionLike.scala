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

package com.salesforce.op.features

import enumeratum._


/**
 * Keeps the distribution information for features
 */
trait FeatureDistributionLike {

  /**
   * name of the feature
   */
  val name: String

  /**
   * map key associated with distribution (when the feature is a map)
   */
  val key: Option[String]

  /**
   * total count of feature seen
   */
  val count: Long

  /**
   * number of empties seen in feature
   */
  val nulls: Long

  /**
   * binned counts of feature values (hashed for strings, evenly spaced bins for numerics)
   */
  val distribution: Array[Double]

  /**
   * either min and max number of tokens for text data, or number of splits used for bins for numeric data
   */
  val summaryInfo: Array[Double]

  /**
   * feature distribution type: training or scoring
   */
  val `type`: FeatureDistributionType

}



/**
 *Feature Distribution Type
 */
sealed trait FeatureDistributionType extends EnumEntry with Serializable

object FeatureDistributionType extends Enum[FeatureDistributionType] {
  val values = findValues
  case object Training extends FeatureDistributionType
  case object Scoring extends FeatureDistributionType
}
