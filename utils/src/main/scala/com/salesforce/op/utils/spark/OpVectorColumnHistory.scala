/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
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
 * @param indicatorGroup The name of the group an indicator belongs to (usually the parent feature, but in the case
 *                       of TextMapVectorizer, this includes keys in maps too). Every other vector column in the same
 *                       vector that has this same indicator group should be mutually exclusive to this one. If
 *                       this is not an indicator, then this field is None
 * @param indicatorValue An indicator for a value (null indicator or result of a pivot or whatever that value is),
 *                       otherwise [[None]]
 * @param index the index of the vector column this information is tied to
 */
case class OpVectorColumnHistory
(
  columnName: String,
  parentFeatureName: Seq[String],
  parentFeatureOrigins: Seq[String],
  parentFeatureStages: Seq[String],
  parentFeatureType: Seq[String],
  indicatorGroup: Option[String],
  indicatorValue: Option[String],
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
