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
    numInvocationsPerThread: Int = 10000,
    atMost: Duration = 10.seconds,
    warmUp: Boolean = true,
    table: TableFor1[T],
    functionCheck: T => Unit
  ): Unit = {
    def doTest(t: TableFor1[T]): Unit = {
      forAll(t) { ft =>
        def testOne(): Unit = {
          var i = 0
          while (i < numInvocationsPerThread) {
            functionCheck(ft)
            i += 1
          }
        }
        val all = Future.sequence((0 until numThreads).map(_ => Future(testOne())))
        val res = Await.result(all, atMost)
        res.length shouldBe numThreads
      }
    }

    def measure(f: => Unit): Long = {
      val started = System.currentTimeMillis()
      val _ = f
      System.currentTimeMillis() - started
    }

    val warmUpElapsed = if (warmUp) Some(measure(doTest(table.take(1)))) else None
    val elapsed = measure(doTest(table))
    println(
      s"Tested with $numThreads concurrent threads. " +
        warmUpElapsed.map(v => s"Warm up: ${v}ms. ").getOrElse("") +
        s"Actual: ${elapsed}ms. " +
        s"Executed ${numInvocationsPerThread * numThreads} function invocations with average " +
        elapsed / (numInvocationsPerThread * numThreads).toDouble +
        "ms per function invocation."
    )
    elapsed should be <= atMost.toMillis
  }

}
