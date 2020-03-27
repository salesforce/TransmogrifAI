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

import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner
import com.salesforce.op.test.{TestCommon, TestSparkContext}

@RunWith(classOf[JUnitRunner])
class JobGroupUtilTest extends FlatSpec with TestCommon with TestSparkContext {

  Spec(JobGroupUtil.getClass) should "be able to set a job group ID around a code block" in {
    JobGroupUtil.withJobGroup(OpStep.DataReadingAndFiltering) {
      spark.sparkContext.parallelize(Seq(1, 2, 3, 4, 5)).collect()
    }
    spark.sparkContext.statusTracker.getJobIdsForGroup("DataReadingAndFiltering") should not be empty
  }

  it should "reset the job group ID after a code block" in {
    JobGroupUtil.withJobGroup(OpStep.DataReadingAndFiltering) {
      spark.sparkContext.parallelize(Seq(1, 2, 3, 4, 5)).collect()
    }
    spark.sparkContext.parallelize(Seq(1, 2, 3, 4, 5)).collect()
    // Ensure that the last `.collect()` was not tagged with "DataReadingAndFiltering"
    spark.sparkContext.statusTracker.getJobIdsForGroup(null) should not be empty
  }
}
