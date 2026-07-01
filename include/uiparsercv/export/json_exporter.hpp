#pragma once

#include "uiparsercv/pipeline/pipeline_runner.hpp"

#include <iosfwd>

namespace uiparsercv::exporter {

void write_json(std::ostream& out, const pipeline::PipelineResult& result);

} // namespace uiparsercv::exporter

