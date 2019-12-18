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
import enumeratum._

// TODO: Make documentation
sealed class SensitiveFeatureInformation(val actionTaken: Boolean = false) extends EnumEntry with JsonLike {
  /**
   * Convert to Spark metadata
   *
   * @return metadata representation
   */
  def toMetadata: Metadata = {
    this match {
      case SensitiveFeatureInformation.Name(actionTaken, probName, genderStrats, probMale, probFemale, probOther) =>
        new MetadataBuilder()
          .putString(SensitiveFeatureInformation.TypeKey, SensitiveFeatureInformation.Name.entryName)
          .putBoolean(SensitiveFeatureInformation.ActionTakenKey, actionTaken)
          .putDouble(SensitiveFeatureInformation.Name.ProbNameKey, probName)
          .putStringArray(SensitiveFeatureInformation.Name.GenderDetectStratsKey, genderStrats.toArray)
          .putDouble(SensitiveFeatureInformation.Name.ProbMaleKey, probMale)
          .putDouble(SensitiveFeatureInformation.Name.ProbFemaleKey, probFemale)
          .putDouble(SensitiveFeatureInformation.Name.ProbOtherKey, probOther)
          .build()
      case _ => throw new RuntimeException(
        "Metadata for sensitive features other than names have not been implemented.")
    }
  }
}
case object SensitiveFeatureInformation extends Enum[SensitiveFeatureInformation] {
  val TypeKey = "DetectedSensitiveFeatureKind"
  val ActionTakenKey = "ActionTaken"
  val values: Seq[SensitiveFeatureInformation] = findValues

  // Utilized by SmartTextVectorizer's name detection
  case class Name
  (
    override val actionTaken: Boolean,
    probName: Double,
    genderDetectionStratsByPerformance: Seq[String],
    probMale: Double,
    probFemale: Double,
    probOther: Double
  ) extends SensitiveFeatureInformation(actionTaken = actionTaken) {
    // TODO: Override toString for more automatic logging
    override def toString: String = super.toString
  }
  case object Name extends SensitiveFeatureInformation {
    override val entryName = "Name"
    val ProbNameKey = "ProbName"
    val GenderDetectStratsKey = "GenderDetectStrats"
    val ProbMaleKey = "ProbMale"
    val ProbFemaleKey = "ProbFemale"
    val ProbOtherKey = "ProbOther"
  }

  // Not yet utilized
  case object Salutation extends SensitiveFeatureInformation
  case object BirthDate extends SensitiveFeatureInformation
  case object PostalCode extends SensitiveFeatureInformation
  case object Other extends SensitiveFeatureInformation

  /**
   * Build metadata from Map of [[SensitiveFeatureInformation]] instances
   *
   * @param map Map from feature name to [[SensitiveFeatureInformation]] of that feature
   * @return metadata representation
   */
  def toMetadata(map: Map[String, SensitiveFeatureInformation]): Metadata = {
    val builder = new MetadataBuilder()
    map.foreach { case (k, v) => builder.putMetadata(k, v.toMetadata) }
    builder.build()
  }

  /**
   * Build Map of [[SensitiveFeatureInformation]] instances from metadata
   *
   * @param meta metadata containing a mapping from feature name to [[SensitiveFeatureInformation]]
   * @return map of that information
   */
  def fromMetadataMap(meta: Metadata): Map[String, SensitiveFeatureInformation] = {
    val infoMap = meta.wrapped.underlyingMap
    infoMap.map { case (k, v) => k -> fromMetadata(v.asInstanceOf[Metadata]) }
  }

  /**
   * Build [[SensitiveFeatureInformation]] from metadata
   *
   * @param meta Metadata representing [[SensitiveFeatureInformation]]
   * @return new instance of [[SensitiveFeatureInformation]]
   */
  def fromMetadata(meta: Metadata): SensitiveFeatureInformation = {
    meta.getString(SensitiveFeatureInformation.TypeKey) match {
      case SensitiveFeatureInformation.Name.entryName =>
        SensitiveFeatureInformation.Name(
          meta.getBoolean(SensitiveFeatureInformation.ActionTakenKey),
          meta.getDouble(SensitiveFeatureInformation.Name.ProbNameKey),
          meta.getStringArray(SensitiveFeatureInformation.Name.GenderDetectStratsKey),
          meta.getDouble(SensitiveFeatureInformation.Name.ProbMaleKey),
          meta.getDouble(SensitiveFeatureInformation.Name.ProbFemaleKey),
          meta.getDouble(SensitiveFeatureInformation.Name.ProbOtherKey)
        )
      case _ => throw new RuntimeException(
        "Metadata for sensitive features other than names have not been implemented.")
    }
  }
}
