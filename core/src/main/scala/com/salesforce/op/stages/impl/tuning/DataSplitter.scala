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

package com.salesforce.op.stages.impl.tuning

import com.salesforce.op.UID
import org.apache.spark.ml.param._
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.types.{Metadata, MetadataBuilder}

case object DataSplitter {

  /**
   * Creates instance that will split data into training and test set
   *
   * @param seed                set for the random split
   * @param reserveTestFraction fraction of the data used for test
   * @return data splitter
   */
  def apply(
    seed: Long = SplitterParamsDefault.seedDefault,
    reserveTestFraction: Double = SplitterParamsDefault.ReserveTestFractionDefault
  ): DataSplitter = {
    new DataSplitter()
      .setSeed(seed)
      .setReserveTestFraction(reserveTestFraction)
  }
}

/**
 * Instance that will split the data into training and holdout for regressions
 *
 * @param uid
 */
class DataSplitter(uid: String = UID[DataSplitter]) extends Splitter(uid = uid) {

  /**
   * Function to set parameters before passing into the validation step
   * eg - do data balancing or dropping based on the labels
   *
   * @param data
   * @return Parameters set in examining data
   */
  override def preValidationPrepare(data: Dataset[Row]): Option[SplitterSummary] = {
    summary = Option(DataSplitterSummary())
    summary
  }

  override def copy(extra: ParamMap): DataSplitter = {
    val copy = new DataSplitter(uid)
    copyValues(copy, extra)
  }
}

/**
 * Empty class because no summary information for a data splitter
 */
case class DataSplitterSummary() extends SplitterSummary {

  /**
   * Converts to [[Metadata]]
   *
   * @param skipUnsupported skip unsupported values
   * @throws RuntimeException in case of unsupported value type
   * @return [[Metadata]] metadata
   */
  def toMetadata(skipUnsupported: Boolean): Metadata = {
    new MetadataBuilder()
      .putString(SplitterSummary.ClassName, this.getClass.getName)
      .build()
  }

}
