import React, { useState } from 'react';
import { UploadCloud, BarChart3, TrendingUp, AlertTriangle, FileBarChart, Loader2 } from 'lucide-react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Upload failed. Ensure backend is running on port 8000.');
      }
      
      const result = await response.json();
      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background text-textMain p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-8">
        
        {/* Header */}
        <header className="flex items-center space-x-4 border-b border-surface pb-6">
          <div className="bg-primary/20 p-3 rounded-xl border border-primary/30 shadow-[0_0_15px_rgba(59,130,246,0.3)]">
            <BarChart3 className="w-8 h-8 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
              GenAI Business Insights
            </h1>
            <p className="text-textMuted mt-1">Upload your sales dataset to generate instant analytics and KPIs.</p>
          </div>
        </header>

        {/* Upload Section */}
        {!data && (
          <section className="bg-surface rounded-2xl p-8 border border-white/5 shadow-xl flex flex-col items-center justify-center min-h-[400px]">
            <form onSubmit={handleUpload} className="w-full max-w-md space-y-6 flex flex-col items-center">
              <label className="w-full group cursor-pointer">
                <div className="border-2 border-dashed border-textMuted/40 rounded-2xl p-12 flex flex-col items-center justify-center transition-all hover:border-primary hover:bg-primary/5">
                  <UploadCloud className="w-16 h-16 text-textMuted group-hover:text-primary transition-colors" />
                  <span className="mt-4 text-lg font-medium text-textMuted group-hover:text-primary">
                    {file ? file.name : "Select CSV Dataset"}
                  </span>
                </div>
                <input 
                  type="file" 
                  accept=".csv" 
                  className="hidden" 
                  onChange={(e) => setFile(e.target.files[0])}
                />
              </label>
              
              <button 
                type="submit"
                disabled={!file || loading}
                className="w-full py-4 px-6 rounded-xl font-semibold flex items-center justify-center space-x-2 transition-all 
                  disabled:opacity-50 disabled:cursor-not-allowed
                  bg-gradient-to-r from-primary to-blue-600 hover:from-blue-600 hover:to-primary text-white shadow-lg hover:shadow-primary/25"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Processing Analytics...</span>
                  </>
                ) : (
                  <>
                    <FileBarChart className="w-5 h-5" />
                    <span>Generate Insights</span>
                  </>
                )}
              </button>
              
              {error && (
                <div className="w-full p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm text-center">
                  {error}
                </div>
              )}
            </form>
          </section>
        )}

        {/* Results Dashboard */}
        {data && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-semibold flex items-center gap-2">
                <TrendingUp className="text-primary w-6 h-6" /> 
                Dashboard Overview
              </h2>
              <button onClick={() => setData(null)} className="px-4 py-2 bg-surface hover:bg-white/5 border border-white/5 rounded-lg text-sm transition-colors">
                Upload New Data
              </button>
            </div>

            {/* KPI Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {data.metrics.map((kpi, i) => (
                <div key={i} className="bg-surface p-6 rounded-2xl border border-white/5 shadow-lg hover:shadow-primary/10 transition-shadow">
                  <p className="text-textMuted text-sm font-medium mb-1">{kpi.name}</p>
                  <p className="text-2xl font-bold text-white">
                    {kpi.unit === 'INR' ? '₹' : ''}
                    {typeof kpi.value === 'number' ? kpi.value.toLocaleString(undefined, {maximumFractionDigits: 2}) : kpi.value}
                    {kpi.unit === 'percent' ? '%' : ''}
                  </p>
                </div>
              ))}
            </div>

            {/* Charts Section */}
            {data.visualizations && data.visualizations.length > 0 && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {data.visualizations.map((viz, idx) => (
                  <div key={idx} className="bg-surface p-6 rounded-2xl border border-white/5 shadow-lg">
                    <h3 className="text-lg font-semibold mb-4">{viz.title}</h3>
                    <div className="h-[300px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        {viz.type === 'line' ? (
                          <LineChart data={viz.data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#2A3441" />
                            <XAxis dataKey={viz.dimension} stroke="#94A3B8" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis stroke="#94A3B8" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `₹${(val/1000)}k`} />
                            <Tooltip contentStyle={{backgroundColor: '#1A2333', borderColor: '#2A3441', borderRadius: '8px', color: '#fff'}} />
                            <Line type="monotone" dataKey="value" stroke="#3B82F6" strokeWidth={3} dot={false} activeDot={{r: 6}} />
                          </LineChart>
                        ) : (
                          <BarChart data={viz.data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#2A3441" vertical={false} />
                            <XAxis dataKey={viz.dimension} stroke="#94A3B8" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis stroke="#94A3B8" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `₹${(val/1000)}k`} />
                            <Tooltip contentStyle={{backgroundColor: '#1A2333', borderColor: '#2A3441', borderRadius: '8px', color: '#fff'}} cursor={{fill: '#2A3441'}} />
                            <Bar dataKey="value" fill="#10B981" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        )}
                      </ResponsiveContainer>
                    </div>
                    <p className="text-sm text-textMuted mt-4 italic">{viz.summary}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Anomalies List */}
            {data.anomalies && data.anomalies.length > 0 && (
              <div className="bg-surface p-6 rounded-2xl border border-white/5 shadow-lg">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <AlertTriangle className="text-yellow-500 w-5 h-5" />
                  Key Anomalies Detected ({data.anomalies.length})
                </h3>
                <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                  {data.anomalies.map((anomaly, i) => (
                    <div key={i} className="bg-[#0B0F19] p-4 rounded-xl border border-white/5 flex justify-between items-center group hover:border-yellow-500/30 transition-colors">
                      <div>
                        <p className="text-sm text-textMuted">Date: <span className="text-white font-medium">{anomaly.observation}</span></p>
                        <p className="text-sm">
                          Detected <span className="font-semibold text-yellow-500">{anomaly.type}</span> {anomaly.metric} 
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-bold text-white">₹{anomaly.value.toLocaleString()}</p>
                        <p className="text-xs text-textMuted">Expected: ₹{anomaly.expected_range.lower.toLocaleString()} - ₹{anomaly.expected_range.upper.toLocaleString()}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        )}

      </div>
    </div>
  );
}
