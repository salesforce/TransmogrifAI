/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package org.apache.spark.sql.execution.datasources.csv

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType

case object CSVSchemaUtils {

  /**
   * Automatically infer CSV schema from the provided RDD. The process is as follows:
   *
   * Similar to the JSON schema inference:
   *  1. Infer type of each row
   *  2. Merge row types to find common type
   *  3. Replace any null types with string type
   *
   * @param rdd     data
   * @param header  CSV header
   * @param options CSV options
   * @return inferred schema
   */
  def infer(
    rdd: RDD[Array[String]],
    header: Seq[String],
    options: com.salesforce.op.utils.io.csv.CSVOptions
  ): StructType = {
    val opts = new org.apache.spark.sql.execution.datasources.csv.CSVOptions(
      parameters = options.copy(header = false).toSparkCSVOptionsMap + ("inferSchema" -> true.toString),
      defaultTimeZoneId = "GMT"
    )
    CSVInferSchema.infer(rdd, header.toArray, opts)
  }

}
