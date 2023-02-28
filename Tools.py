from plotly.graph_objects import Figure, Heatmap

class Tools:
    def heatmap_fig(z, x = None, y = None, title = None, hover_text= None) -> Figure:
        heatmap = Heatmap(
            z = z, 
            hoverinfo= "text" if type(hover_text) != type(None) else None, 
            text=hover_text,
            )
        
        fig = Figure(data=heatmap)
        fig.update_layout(title = title)
        return fig