/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
