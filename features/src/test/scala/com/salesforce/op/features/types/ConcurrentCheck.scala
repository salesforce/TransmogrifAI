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

package com.salesforce.op.features.types

import org.scalatest.Matchers
import org.scalatest.prop.{PropertyChecks, TableFor1}

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}


trait ConcurrentCheck {
  self: PropertyChecks with Matchers =>

  def forAllConcurrentCheck[T](
    numThreads: Int = 10,
    numInstancesPerThread: Int = 50000,
    atMost: Duration = 10.seconds,
    table: TableFor1[T],
    functionCheck: T => Unit
  ): Unit = {

    val started = System.currentTimeMillis()
    forAll(table) { ft =>
      def testOne() = {
        var i = 0
        while (i < numInstancesPerThread) {
          functionCheck(ft)
          i += 1
        }
      }

      val all = Future.sequence((0 until numThreads).map(_ => Future(testOne())))
      val res = Await.result(all, atMost)
      res.length shouldBe numThreads
    }
    val elapsed = System.currentTimeMillis() - started
    println(
      s"Tested with $numThreads concurrent threads. Elapsed: ${elapsed}ms. " +
        s"Created ${numInstancesPerThread * numThreads} instances with average " +
        (System.currentTimeMillis() - started) / (numInstancesPerThread * numThreads).toDouble +
        "ms per instance creation."
    )
    elapsed should be <= atMost.toMillis
  }

}
