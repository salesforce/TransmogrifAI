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

package com.salesforce.op

import com.salesforce.op.utils.json.JsonLike
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}

/**
 * A base class for different SensitiveFeatureInformation
 * The following three params are required for every kind of SensitiveFeatureInformation
 *
 * @param name        the name of the raw feature
 * @param key         optionally, the name of the key (if the raw feature is a Map type)
 * @param actionTaken whether the handling of the raw feature changed b/c it was detected as sensitive
 */
sealed abstract class SensitiveFeatureInformation
(
  val name: String,
  val key: Option[String] = None,
  val actionTaken: Boolean = false
) extends JsonLike {
  val EntryName: String
  def toMetadata: Metadata
}

object SensitiveFeatureInformation {
  val NameKey = "FeatureName"
  val MapKeyKey = "MapKey"
  val ActionTakenKey = "ActionTaken"
  val TypeKey = "DetectedSensitiveFeatureKind"

  /**
   * Build metadata from Map of [[SensitiveFeatureInformation]] instances
   *
   * @param map Map from feature name to Seq of [[SensitiveFeatureInformation]] about that feature
   * @return metadata representation
   */
  def toMetadata(map: Map[String, Seq[SensitiveFeatureInformation]]): Metadata = {
    val builder = new MetadataBuilder()
    map.foreach { case (k, values) => builder.putMetadataArray(k, values map { _.toMetadata } toArray) }
    builder.build()
  }

  /**
   * Build Map of [[SensitiveFeatureInformation]] instances from metadata
   *
   * @param meta metadata containing a mapping from feature name to [[SensitiveFeatureInformation]]
   * @return map of that information
   */
  def fromMetadataMap(meta: Metadata): Map[String, Seq[SensitiveFeatureInformation]] = {
    val infoMap = meta.wrapped.underlyingMap
    infoMap.map { case (k, values) => k -> values.asInstanceOf[Array[Metadata]].map(fromMetadata).toSeq }
  }

  /**
   * Build [[SensitiveFeatureInformation]] from metadata
   *
   * @param meta Metadata representing [[SensitiveFeatureInformation]]
   * @return new instance of [[SensitiveFeatureInformation]]
   */
  def fromMetadata(meta: Metadata): SensitiveFeatureInformation = {
    meta.getString(SensitiveFeatureInformation.TypeKey) match {
      case SensitiveNameInformation.EntryName =>
        SensitiveNameInformation(
          meta.getDouble(SensitiveNameInformation.ProbNameKey),
          meta.getStringArray(SensitiveNameInformation.GenderDetectStratsKey),
          meta.getDouble(SensitiveNameInformation.ProbMaleKey),
          meta.getDouble(SensitiveNameInformation.ProbFemaleKey),
          meta.getDouble(SensitiveNameInformation.ProbOtherKey),
          meta.getString(SensitiveFeatureInformation.NameKey),
          {
            val mapKey = meta.getString(SensitiveFeatureInformation.MapKeyKey)
            if (mapKey.isEmpty) None else Some(mapKey)
          },
          meta.getBoolean(SensitiveFeatureInformation.ActionTakenKey)
        )
      case _ => throw new RuntimeException(
        "Metadata for sensitive features other than names have not been implemented.")
    }
  }
}

case class SensitiveNameInformation
(
  probName: Double,
  genderDetectResults: Seq[String],
  probMale: Double,
  probFemale: Double,
  probOther: Double,
  override val name: String,
  override val key: Option[String] = None,
  override val actionTaken: Boolean = false
) extends SensitiveFeatureInformation(name, key, actionTaken) {
  override val EntryName: String = SensitiveNameInformation.EntryName
  override def toMetadata: Metadata = {
    new MetadataBuilder()
      .putString(SensitiveFeatureInformation.NameKey, name)
      .putString(SensitiveFeatureInformation.MapKeyKey, key.getOrElse(""))
      .putBoolean(SensitiveFeatureInformation.ActionTakenKey, actionTaken)
      .putString(SensitiveFeatureInformation.TypeKey, this.EntryName)
      .putDouble(SensitiveNameInformation.ProbNameKey, probName)
      .putStringArray(SensitiveNameInformation.GenderDetectStratsKey, genderDetectResults.toArray)
      .putDouble(SensitiveNameInformation.ProbMaleKey, probMale)
      .putDouble(SensitiveNameInformation.ProbFemaleKey, probFemale)
      .putDouble(SensitiveNameInformation.ProbOtherKey, probOther)
      .build()
  }
}

case object SensitiveNameInformation {
  val EntryName = "SensitiveNameInformation"
  val ProbNameKey = "ProbName"
  val GenderDetectStratsKey = "GenderDetectStrats"
  val ProbMaleKey = "ProbMale"
  val ProbFemaleKey = "ProbFemale"
  val ProbOtherKey = "ProbOther"
}

// TODO: Use this everywhere
case class GenderDetectionResults(strategyString: String, pctUnidentified: Double) extends JsonLike
