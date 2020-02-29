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

package com.salesforce.op.stages.impl.preparators

import com.salesforce.op.stages.impl.MetadataLike
import com.salesforce.op.utils.spark.RichMetadata._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}

import scala.util.{Failure, Success, Try}


/**
 * Case class to store metadata from [[MinVarianceFilter]]
 *
 * @param dropped            features dropped by minimum variance filter
 * @param featuresStatistics stats on features
 * @param names              names of features passed in
 */
case class MinVarianceSummary
(
  dropped: Seq[String],
  featuresStatistics: SummaryStatistics,
  names: Seq[String]
) extends MetadataLike {

  /**
   * Converts to [[Metadata]]
   *
   * @param skipUnsupported skip unsupported values
   * @throws RuntimeException in case of unsupported value type
   * @return [[Metadata]] metadata
   */
  def toMetadata(skipUnsupported: Boolean): Metadata = {
    val summaryMeta = new MetadataBuilder()
    summaryMeta.putStringArray(MinVarianceNames.Dropped, dropped.toArray)
    summaryMeta.putMetadata(MinVarianceNames.FeaturesStatistics, featuresStatistics.toMetadata(skipUnsupported))
    summaryMeta.putStringArray(MinVarianceNames.Names, names.toArray)
    summaryMeta.build()
  }
}

case object MinVarianceNames extends DerivedFeatureFilterNames

case object MinVarianceSummary extends DerivedFeatureFilterSummary {
  /**
   * Converts metadata into instance of MinVarianceSummary
   *
   * @param meta metadata produced by [[MinVarianceFilter]] which contains summary information
   * @return an instance of the [[MinVarianceSummary]]
   */
  def fromMetadata(meta: Metadata): MinVarianceSummary = {
    val wrapped = meta.wrapped
    Try {
      MinVarianceSummary(
        dropped = wrapped.getArray[String](MinVarianceNames.Dropped).toSeq,
        featuresStatistics = statisticsFromMetadata(wrapped.get[Metadata](MinVarianceNames.FeaturesStatistics)),
        names = wrapped.getArray[String](MinVarianceNames.Names).toSeq
      )
    } match {
      case Success(summary) => summary
      case Failure(_) => throw new IllegalArgumentException(s"failed to parse MinVarianceSummary from $meta")
    }
  }
}
