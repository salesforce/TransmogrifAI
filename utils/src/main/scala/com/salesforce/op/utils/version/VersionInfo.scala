/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.version

import java.util.Properties

import com.salesforce.op.utils.json.JsonLike

import scala.collection.JavaConverters._
import scala.util.Try

/**
 * Container with project version and git related information
 *
 * @param version       project version
 * @param buildTime     build time of this jar
 * @param gitRepo       git repo
 * @param gitBranch     git branch
 * @param gitCommitId   git commit id
 * @param gitCommitTime git commit time
 */
case class VersionInfo
(
  version: Option[String],
  buildTime: Option[String],
  gitRepo: Option[String],
  gitBranch: Option[String],
  gitCommitId: Option[String],
  gitCommitTime: Option[String]
) extends JsonLike

object VersionInfo {

  /**
   * Expecting a properties file to be present as a resource and of the following structure:
   *   version=1.2.3
   *   build.time=2018-01-26 14:02:40 -0800
   *   git.repo=git@github.com:foo/bar.git
   *   git.branch=master
   *   git.commit.id=08b2d82ad212e05f15af383340b94be9af973cd5
   *   git.commit.time=Fri Jan 26 11:21:13 2018 -0800
   */
  private lazy val versionPropsMap: Map[String, String] = Try {
    val prop = new Properties()
    prop.load(this.getClass.getClassLoader.getResourceAsStream("op-version.properties"))
    prop.asScala.toMap
  }.getOrElse(Map.empty)

  private lazy val versionInfo = VersionInfo(
    version = versionPropsMap.get("version"),
    buildTime = versionPropsMap.get("build.time"),
    gitRepo = versionPropsMap.get("git.repo"),
    gitBranch = versionPropsMap.get("git.branch"),
    gitCommitId = versionPropsMap.get("git.commit.id"),
    gitCommitTime = versionPropsMap.get("git.commit.time")
  )

  /**
   * Get current VersionInfo
   */
  def apply(): VersionInfo = versionInfo

}
