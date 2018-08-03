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

package com.salesforce.op.stages.impl.insights

import com.salesforce.op.features.types.TextMap
import com.salesforce.op.utils.json.JsonUtils
import com.salesforce.op.utils.spark.OpVectorColumnHistory
import org.json4s.{DefaultFormats, ShortTypeHints}
import org.json4s.jackson.JsonMethods.parse
import org.json4s.jackson.Serialization
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.{write => jwrite}

/**
 * Converts record insights format Map[String, String] to and from underlying case classes to give
 * Map[OpVectorColumnHistory, Seq[(Int, Double)] the OpVectorColumnHistory contains all information
 * about the construction and value of the feature the insight is for the seq contains the index
 * (from prediction vector) for the insight and the value of the the insight (value can be correlation based
 * or change in score)
 */
object RecordInsightsParser {

  /**
   * All insights take the for sequence of tuples from index of prediction explained to importance value
   */
  type Insights = Seq[(Int, Double)]

  /**
   * Convert insight into strings
   * @param columnInfo OpVectorColumnHistory already in json
   * @param scoreDiffs Score values for record insights
   * @return strings for insights map
   */
  def insightToText(columnInfo: String, scoreDiffs: Array[Double]): (String, String) = {
    val scores = scoreDiffs.zipWithIndex.map{ case (s, i) => Array(i, s) }
    columnInfo -> JsonUtils.toJsonString(scores, pretty = false)
  }

  /**
   * Convert insight into strings
   * @param insight OpVectorColumnHistory and score values ziped with index
   * @return strings for insights map
   */
  def insightToText(insight: (OpVectorColumnHistory, Insights)): (String, String) = {
    val scores = insight._2.map{ case (i, s) => Array(i, s) }
    insight._1.toJson(false) -> JsonUtils.toJsonString(scores, pretty = false)
  }

  /**
   * Takes sting version of insights and converts back to classes for easy parsing
   * @param insights Map[String, String] containing record insights
   * @return Map with OpVectorColumnHistory to insights sequence of values
   */
  def parseInsights(insights: TextMap): Map[OpVectorColumnHistory, Insights] = {
    implicit val formats: DefaultFormats = DefaultFormats
    insights.value.map { case (k, v) => OpVectorColumnHistory.fromJson(k) ->
      parse(v).extract[Seq[Seq[Double]]].map( s => s.head.toInt -> s(1)) }
  }
}
