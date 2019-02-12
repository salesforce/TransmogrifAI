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

import com.salesforce.op.utils.json.JsonLike
import org.json4s._
import org.json4s.jackson.JsonMethods._

/**
 * Full history for each column element in a vector
 *
 * @param columnName name for feature in column
 * @param parentFeatureName name of immediate parent feature that was used to create the vector
 * @param parentFeatureOrigins names of raw features that went into the parent feature
 * @param parentFeatureStages stageNames of all stages applied to the parent feature before conversion to a vector
 * @param parentFeatureType type of the parent feature
 * @param grouping The name of the group a column belongs to (usually the parent feature, but in the case
 *                       of TextMapVectorizer, this includes keys in maps too). Every other vector column in the same
 *                       vector that has this same indicator group should be mutually exclusive to this one. If
 *                       this is not an indicator, then this field is None
 * @param indicatorValue A name for a binary indicator value (null indicator or result of a pivot or whatever that
 *                       value is), otherwise [[None]]
 * @param descriptorValue   A name for a value that is continuous (not a binary indicator) eg for geolocation (lat,
 *                          lon, accuracy) or for dates that have been converted to a circular representation the time
 *                          window and x or y coordinate, otherwise [[None]]
 * @param index the index of the vector column this information is tied to
 */
case class OpVectorColumnHistory
(
  columnName: String,
  parentFeatureName: Seq[String],
  parentFeatureOrigins: Seq[String],
  parentFeatureStages: Seq[String],
  parentFeatureType: Seq[String],
  grouping: Option[String],
  indicatorValue: Option[String],
  descriptorValue: Option[String],
  index: Int
) extends JsonLike

case object OpVectorColumnHistory {

  /**
   * Read vector column history from a json
   *
   * @param json vector column history in json
   * @return Try[OpVectorColumnHistory]
   */
  def fromJson(json: String): OpVectorColumnHistory = {
    implicit val formats: DefaultFormats = DefaultFormats
    parse(json).extract[OpVectorColumnHistory]
  }

}
