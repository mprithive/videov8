import React, { useState, useEffect } from 'react';
import './MemoryWidget.css';

function MemoryWidget() {
  const [memoryData, setMemoryData] = useState({
    usedJSHeapSize: 0,
    totalJSHeapSize: 0,
    jsHeapSizeLimit: 0,
  });

  useEffect(() => {
    const updateMemory = () => {
      if (performance.memory) {
        setMemoryData({
          usedJSHeapSize: Math.round(performance.memory.usedJSHeapSize / 1048576), // Convert to MB
          totalJSHeapSize: Math.round(performance.memory.totalJSHeapSize / 1048576),
          jsHeapSizeLimit: Math.round(performance.memory.jsHeapSizeLimit / 1048576),
        });
      }
    };

    updateMemory();
    const interval = setInterval(updateMemory, 500); // Update every 500ms

    return () => clearInterval(interval);
  }, []);

  const usagePercentage = (memoryData.usedJSHeapSize / memoryData.jsHeapSizeLimit) * 100;

  return (
    <div className="memory-widget">
      <div className="memory-header">Memory</div>
      <div className="memory-content">
        <div className="memory-used">
          <span className="label">Used:</span>
          <span className="value">{memoryData.usedJSHeapSize} MB</span>
        </div>
        <div className="memory-limit">
          <span className="label">Limit:</span>
          <span className="value">{memoryData.jsHeapSizeLimit} MB</span>
        </div>
        <div className="memory-bar">
          <div
            className="memory-bar-fill"
            style={{
              width: `${Math.min(usagePercentage, 100)}%`,
              backgroundColor: usagePercentage > 80 ? '#ff6b6b' : usagePercentage > 50 ? '#ffd93d' : '#7fd8be',
            }}
          ></div>
        </div>
        <div className="memory-percentage">{Math.round(usagePercentage)}%</div>
      </div>
    </div>
  );
}

export default MemoryWidget;
